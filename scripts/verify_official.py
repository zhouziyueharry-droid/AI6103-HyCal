"""Numerically compare cached Part A with the unmodified, pinned HyCal source.

Only train support and validation queries are selected. No test scoring occurs.
Run: .venv/Scripts/python.exe scripts/verify_official.py
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get('HYCAL_OFFICIAL_CHECKOUT', ROOT / 'external/HyCal-CIL')).resolve()
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import clip
from PIL import Image
import torch
from torch.utils.data import DataLoader
from hycal.hycal_model import HyCalModel
from hycal.hycal_evaluator import HyCalEvaluator
import hycal_core as core
from run_experiments import load_inputs, prepare_stats, invert

COMMIT = '4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41'
OUTPUT = ROOT / 'results/part_a/reference_verification.json'
# Declared before comparisons. Precision max error is also scaled against its
# matrix norm: inversion can magnify float32 covariance reduction differences.
LIMITS = {
    'embedding_max_abs': 1e-5,
    'prototype_max_abs': 2e-6,
    'covariance_max_abs': 2e-7,
    'precision_max_abs': 0.25,
    'precision_relative_frobenius': 1e-4,
    'score_max_abs': 2e-4,
    'prediction_agreement': 1.0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(result):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


def errors(actual, expected):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    difference = actual - expected
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        'max_abs': difference.abs().max().item(),
        'mean_abs': difference.abs().mean().item(),
        'relative_frobenius': (difference.norm() / expected.norm().clamp_min(1e-15)).item(),
        'finite': finite,
    }


def within(error, limit, relative_limit=None):
    return error['finite'] and error['max_abs'] <= limit and (
        relative_limit is None or error['relative_frobenius'] <= relative_limit)


class LabelledTensors:
    def __init__(self, tensors, labels):
        self.tensors, self.labels = tensors, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.tensors[index], self.labels[index]


class RawFeatureAdapter:
    """Bypass only the encoder with raw outputs from a real official CLIP run.

    Official image/text normalization, query fusion, and scoring remain intact.
    Token rows are matched, so upstream candidate order is preserved.
    """
    def __init__(self, tokens, text_raw):
        self.text_lookup = {tuple(token.tolist()): feature for token, feature in zip(tokens.cpu(), text_raw.cpu())}

    def encode_image(self, images):
        return images

    def encode_text(self, tokens):
        return torch.stack([self.text_lookup[tuple(token.tolist())] for token in tokens.cpu()]).to(tokens.device)


def official_model(cfg, encoder, device):
    model = HyCalModel.__new__(HyCalModel)
    model.device = device
    model.model = encoder
    model.tokenizer = clip.tokenize
    model.context_length = 77
    model.template = [cfg['template']]
    model.fusion = 'sum'
    model.bg_ref_class = 'first'
    model.sim_metric = 'weighted'
    model.n_min_p = cfg['n_min_p']
    model.scale = cfg['scale']
    model.lambda_reg = cfg['lambda_reg']
    model.lambda_bg = cfg['lambda_reg']
    model.real = False
    model.gamma = 1.0
    model.class_stats = {}
    model.learned_classes = []
    model.bg_stats = {'mu_bg': None, 'P_bg': None, 'N_total': 0}
    return model


def evaluate_capture(model, tensors, labels, classnames):
    evaluator = HyCalEvaluator(model)
    normalized, raw, final = [], [], []
    original_normalize = evaluator._normalize_rmd_scores
    original_argmax = torch.argmax

    def capture_normalize(scores):
        value = original_normalize(scores)
        raw.append(scores.detach().cpu().clone())
        normalized.append(value.detach().cpu().clone())
        return value

    def capture_argmax(scores, *args, **kwargs):
        if scores.ndim == 2 and scores.shape[1] == len(classnames):
            final.append(scores.detach().cpu().clone())
        return original_argmax(scores, *args, **kwargs)

    evaluator._normalize_rmd_scores = capture_normalize
    loader = DataLoader(LabelledTensors(tensors, labels), batch_size=len(labels), shuffle=False)
    with patch('torch.argmax', side_effect=capture_argmax):
        accuracy, _ = evaluator.evaluate(loader, target_classes=classnames)
    assert len(normalized) == 2 and len(raw) == 2 and len(final) == 1
    return {'cosine': normalized[0], 'md': normalized[1], 'h0': final[0],
            'cosine_raw': raw[0], 'md_raw': raw[1], 'accuracy': accuracy}


def score_comparison(local_cos, local_md, local_h0, reference):
    pairs = {'cosine': local_cos, 'md': local_md, 'h0': local_h0}
    comparison = {}
    for name, local in pairs.items():
        e = errors(local, reference[name])
        agreement = (local.argmax(1).cpu() == reference[name].argmax(1)).float().mean().item()
        comparison[name] = {'errors': e, 'prediction_agreement': agreement,
                            'passed': within(e, LIMITS['score_max_abs']) and agreement == 1.0}
    comparison['passed'] = all(comparison[name]['passed'] for name in pairs)
    return comparison


def main():
    started = time.perf_counter()
    result = {'passed': False, 'sourcecommit': COMMIT, 'tolerances': LIMITS,
              'scope': {'seed': 0, 'shots': [1, 5, 10], 'candidate_count': 102,
                        'statistics_device': 'cpu', 'test_split_scored': False,
                        'test_images_opened': False,
                        'statistics': 'actual upstream compute_class_statistics and update_background_stats',
                        'scoring': 'actual upstream HyCalEvaluator.evaluate; raw CLIP feature adapter on CPU and actual CLIP on CUDA',
                        'query_selection': 'first 32 validation manifest rows',
                        'balanced_query_selection': 'first validation row per class, all 102 classes',
                        'balanced_adapter': 'normalized cached images/text are encoder outputs; upstream normalizes them again; isolated comparison uses those same renormalized inputs',
                        'support_order': 'same fixed support manifest order in both implementations'},
              'cases': []}
    save(result)
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        head = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(OFFICIAL), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        assert head == COMMIT and not status, (head, status)
        cfg, cache, records, supports = load_inputs()
        classes = cache['classes']
        classnames = [c['classname'] for c in classes]
        assert classnames == sorted(classnames) and len(classnames) == 102
        assert cache['image_features'].dtype == torch.float32
        val_indices = [i for i, row in enumerate(records) if row['split'] == 'val'][:32]
        assert len(val_indices) == 32
        val_records = [records[i] for i in val_indices]
        names_by_label = {c['label']: c['classname'] for c in classes}
        val_labels = [names_by_label[row['label']] for row in val_records]
        first_validation_by_label = {}
        for index, row in enumerate(records):
            if row['split'] == 'val':
                first_validation_by_label.setdefault(row['label'], index)
        balanced_indices = [first_validation_by_label[c['label']] for c in classes]
        assert len(balanced_indices) == 102
        balanced_images = cache['image_features'][balanced_indices]
        balanced_labels = classnames
        result['scope']['balanced_validation_sample_ids'] = [records[i]['sample_id'] for i in balanced_indices]
        result['scope']['validation_sample_ids'] = [row['sample_id'] for row in val_records]
        result['provenance'] = {'official_head': head, 'official_tracked_clean': not status,
            'torch': torch.__version__, 'core_sha256': sha(ROOT/'src/hycal_core.py'),
            'runner_sha256': sha(ROOT/'scripts/run_experiments.py'),
            'config_sha256': sha(ROOT/'configs/part_a.json'),
            'feature_sha256': sha(ROOT/'artifacts/features/flowers102/features.pt'),
            'support_sha256': sha(ROOT/'data/manifests/flowers102/support_indices.json'),
            'official_model_sha256': sha(OFFICIAL/'hycal/hycal_model.py'),
            'official_evaluator_sha256': sha(OFFICIAL/'hycal/hycal_evaluator.py')}

        # Re-encode actual validation image pixels with the actual official fork.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        result['scope']['statistics_devices'] = ['cpu', 'cuda'] if device == 'cuda' else ['cpu']
        backbone, _, preprocess = clip.load(str(ROOT/'artifacts/models/ViT-B-16.pt'), device=device, jit=False)
        backbone.eval().requires_grad_(False)
        assert next(backbone.parameters()).dtype == torch.float32
        pixel_batches = []
        for row in val_records:
            assert row['split'] == 'val'
            with Image.open(row['path']) as image:
                pixel_batches.append(preprocess(image))
        pixels = torch.stack(pixel_batches)
        prompts = [cfg['template'].format(name.replace('_', ' ')) for name in classnames]
        tokens = clip.tokenize(prompts)
        with torch.inference_mode():
            raw_images = backbone.encode_image(pixels.to(device)).cpu()
            raw_text = backbone.encode_text(tokens.to(device)).cpu()
            actual_images = HyCalModel._normalize_embedding(None, raw_images)
            actual_text = HyCalModel._normalize_embedding(None, raw_text)
        image_error = errors(actual_images, cache['image_features'][val_indices])
        text_error = errors(actual_text, cache['text_features'])
        result['cache_reencoding'] = {'device': device, 'dtype': 'float32',
            'image_count': 32, 'text_count': 102, 'image_errors': image_error, 'text_errors': text_error,
            'passed': within(image_error, LIMITS['embedding_max_abs']) and within(text_error, LIMITS['embedding_max_abs'])}
        print('CLIP cache re-encoding ' + json.dumps(result['cache_reencoding']), flush=True)
        adapter = RawFeatureAdapter(tokens, raw_text)
        lookup = {row['sample_id']: i for i, row in enumerate(records)}
        candidate_by_label = {c['label']: i for i, c in enumerate(classes)}

        for shot in [1, 5, 10]:
            case_started = time.perf_counter()
            reference_model = official_model(cfg, adapter, 'cpu')
            support_ids = supports['indices']['0'][str(shot)]
            indices = [lookup[sample_id] for sample_id in support_ids]
            assert all(records[index]['split'] == 'train' for index in indices)
            support_labels = [names_by_label[records[index]['label']] for index in indices]
            support_text = cache['text_features'][[candidate_by_label[records[index]['label']] for index in indices]]
            fused = reference_model._normalize_embedding(cache['image_features'][indices] + support_text)
            captured_covariances = []
            original_precision = reference_model._compute_regularized_precision
            def capture_precision(cov, *args, **kwargs):
                captured_covariances.append(cov.detach().clone())
                return original_precision(cov, *args, **kwargs)
            reference_model._compute_regularized_precision = capture_precision
            # These are genuine methods imported from the unchanged source.
            official_stats = reference_model.compute_class_statistics(fused, support_labels)
            reference_model.update_stats(official_stats)
            reference_model.update_background_stats(fused)
            assert reference_model.learned_classes == classnames
            ref_proto = torch.stack([official_stats[name][0] for name in classnames])
            ref_precision = torch.stack([official_stats[name][1] for name in classnames])
            local_proto, local_cov, local_text, (local_bg_mu, local_bg_cov) = prepare_stats(
                cache, records, supports, 0, shot, 'cpu')
            local_precision = invert(core.regularize_covariance(local_cov, 0.0, cfg['lambda_reg']))
            local_bg_precision = torch.pinverse(local_bg_cov + cfg['lambda_reg'] * torch.eye(local_bg_cov.shape[0]))
            prototype_error = errors(local_proto, ref_proto)
            covariance_error = errors(local_cov, torch.stack(captured_covariances[:102]))
            precision_error = errors(local_precision, ref_precision)
            bg_mean_error = errors(local_bg_mu, reference_model.bg_stats['mu_bg'])
            bg_precision_error = errors(local_bg_precision, reference_model.bg_stats['P_bg'])
            statistics = {'prototype': prototype_error, 'covariance': covariance_error, 'precision': precision_error,
                          'background_mean': bg_mean_error, 'background_precision': bg_precision_error}
            statistics['passed'] = (within(prototype_error, LIMITS['prototype_max_abs']) and
                within(covariance_error, LIMITS['covariance_max_abs']) and
                within(precision_error, LIMITS['precision_max_abs'], LIMITS['precision_relative_frobenius']) and
                within(bg_mean_error, LIMITS['prototype_max_abs']) and
                within(bg_precision_error, LIMITS['precision_max_abs'], LIMITS['precision_relative_frobenius']))

            reference = evaluate_capture(reference_model, raw_images, val_labels, classnames)
            local_cos, local_md = core.score_features(cache['image_features'][val_indices], local_text,
                local_proto, local_precision, 32, (local_bg_mu, local_bg_precision))
            counts = torch.full((102,), shot)
            local_h0 = core.fuse_scores(local_cos, local_md, counts, 'H0', cfg['n_min_p'], cfg['scale'])
            cache_comparison = score_comparison(local_cos, local_md, local_h0, reference)

            # Isolate scoring code from independently recomputed/cache errors by
            # using the exact official statistics and freshly normalized inputs.
            exact_cos, exact_md = core.score_features(actual_images, actual_text, ref_proto, ref_precision, 32,
                (reference_model.bg_stats['mu_bg'], reference_model.bg_stats['P_bg']))
            exact_h0 = core.fuse_scores(exact_cos, exact_md, counts, 'H0', cfg['n_min_p'], cfg['scale'])
            pure_scoring = score_comparison(exact_cos, exact_md, exact_h0, reference)

            balanced_adapter = RawFeatureAdapter(tokens, cache['text_features'])
            reference_model.model = balanced_adapter
            balanced_reference = evaluate_capture(reference_model, balanced_images, balanced_labels, classnames)
            balanced_cos, balanced_md = core.score_features(balanced_images, local_text, local_proto, local_precision,
                102, (local_bg_mu, local_bg_precision))
            balanced_h0 = core.fuse_scores(balanced_cos, balanced_md, counts, 'H0', cfg['n_min_p'], cfg['scale'])
            balanced_comparison = score_comparison(balanced_cos, balanced_md, balanced_h0, balanced_reference)
            balanced_exact_cos, balanced_exact_md = core.score_features(reference_model._normalize_embedding(balanced_images),
                reference_model._normalize_embedding(cache['text_features']), ref_proto, ref_precision, 102,
                (reference_model.bg_stats['mu_bg'], reference_model.bg_stats['P_bg']))
            balanced_exact_h0 = core.fuse_scores(balanced_exact_cos, balanced_exact_md, counts, 'H0', cfg['n_min_p'], cfg['scale'])
            balanced_pure_scoring = score_comparison(balanced_exact_cos, balanced_exact_md, balanced_exact_h0, balanced_reference)

            # The actual upstream evaluator also consumes real image pixels and
            # the actual CLIP instance, independently of the feature adapter.
            real_model = official_model(cfg, backbone, device)
            real_model.class_stats = {name: (mu.to(device), p.to(device), n)
                                      for name, (mu, p, n) in official_stats.items()}
            real_model.learned_classes = classnames
            real_model.bg_stats = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                                   for key, value in reference_model.bg_stats.items()}
            actual_reference = evaluate_capture(real_model, pixels, val_labels, classnames)
            actual_cos, actual_md = core.score_features(cache['image_features'][val_indices].to(device),
                local_text.to(device), local_proto.to(device), local_precision.to(device), 32,
                (local_bg_mu.to(device), local_bg_precision.to(device)))
            actual_h0 = core.fuse_scores(actual_cos, actual_md, counts, 'H0', cfg['n_min_p'], cfg['scale'])
            real_comparison = score_comparison(actual_cos, actual_md, actual_h0, actual_reference)
            gpu_complete = None
            if device == 'cuda':
                gpu_model = official_model(cfg, backbone, device)
                gpu_fused = gpu_model._normalize_embedding(cache['image_features'][indices].to(device) + support_text.to(device))
                gpu_covariances = []
                gpu_precision_function = gpu_model._compute_regularized_precision
                def capture_gpu_precision(cov, *args, **kwargs):
                    gpu_covariances.append(cov.detach().clone())
                    return gpu_precision_function(cov, *args, **kwargs)
                gpu_model._compute_regularized_precision = capture_gpu_precision
                gpu_stats = gpu_model.compute_class_statistics(gpu_fused, support_labels)
                gpu_model.update_stats(gpu_stats)
                gpu_model.update_background_stats(gpu_fused)
                gpu_ref_proto = torch.stack([gpu_stats[name][0] for name in classnames])
                gpu_ref_precision = torch.stack([gpu_stats[name][1] for name in classnames])
                gpu_proto, gpu_cov, gpu_text, (gpu_bg_mu, gpu_bg_cov) = prepare_stats(cache, records, supports, 0, shot, device)
                gpu_precision = invert(core.regularize_covariance(gpu_cov, 0.0, cfg['lambda_reg']))
                gpu_bg_precision = torch.pinverse(gpu_bg_cov + cfg['lambda_reg'] * torch.eye(gpu_bg_cov.shape[0], device=device))
                gpu_errors = {'prototype': errors(gpu_proto, gpu_ref_proto),
                    'covariance': errors(gpu_cov, torch.stack(gpu_covariances[:102])),
                    'precision': errors(gpu_precision, gpu_ref_precision),
                    'background_mean': errors(gpu_bg_mu, gpu_model.bg_stats['mu_bg']),
                    'background_precision': errors(gpu_bg_precision, gpu_model.bg_stats['P_bg'])}
                gpu_errors['passed'] = (within(gpu_errors['prototype'], LIMITS['prototype_max_abs']) and
                    within(gpu_errors['covariance'], LIMITS['covariance_max_abs']) and
                    within(gpu_errors['precision'], LIMITS['precision_max_abs'], LIMITS['precision_relative_frobenius']) and
                    within(gpu_errors['background_mean'], LIMITS['prototype_max_abs']) and
                    within(gpu_errors['background_precision'], LIMITS['precision_max_abs'], LIMITS['precision_relative_frobenius']))
                gpu_reference = evaluate_capture(gpu_model, pixels, val_labels, classnames)
                gpu_cos, gpu_md = core.score_features(cache['image_features'][val_indices].to(device), gpu_text,
                    gpu_proto, gpu_precision, 32, (gpu_bg_mu, gpu_bg_precision))
                gpu_h0 = core.fuse_scores(gpu_cos, gpu_md, counts, 'H0', cfg['n_min_p'], cfg['scale'])
                gpu_scores = score_comparison(gpu_cos, gpu_md, gpu_h0, gpu_reference)
                gpu_model.model = balanced_adapter
                gpu_balanced_reference = evaluate_capture(gpu_model, balanced_images, balanced_labels, classnames)
                gpu_balanced_cos, gpu_balanced_md = core.score_features(balanced_images.to(device), gpu_text,
                    gpu_proto, gpu_precision, 102, (gpu_bg_mu, gpu_bg_precision))
                gpu_balanced_h0 = core.fuse_scores(gpu_balanced_cos, gpu_balanced_md, counts, 'H0', cfg['n_min_p'], cfg['scale'])
                gpu_balanced_scores = score_comparison(gpu_balanced_cos, gpu_balanced_md, gpu_balanced_h0, gpu_balanced_reference)
                gpu_complete = {'statistics': gpu_errors, 'scores': gpu_scores,
                                'balanced_102_query_scores': gpu_balanced_scores,
                                'passed': gpu_errors['passed'] and gpu_scores['passed'] and gpu_balanced_scores['passed']}
            case = {'shot': shot, 'seed': 0, 'candidate_count': 102, 'query_count': 32,
                    'statistics': statistics, 'cached_pipeline_vs_official_cpu': cache_comparison,
                    'same_inputs_scoring_vs_official_cpu': pure_scoring,
                    'balanced_102_query_cached_pipeline_vs_official_cpu': balanced_comparison,
                    'balanced_102_query_same_inputs_scoring_vs_official_cpu': balanced_pure_scoring,
                    'cached_pipeline_vs_actual_clip_evaluator': real_comparison,
                    'full_cuda_statistics_and_actual_evaluator': gpu_complete,
                    'h0_equals_cosine_below_gate': torch.equal(local_h0, local_cos) if shot < cfg['n_min_p'] else None,
                    'seconds': time.perf_counter() - case_started,
                    'passed': statistics['passed'] and cache_comparison['passed'] and
                              pure_scoring['passed'] and real_comparison['passed'] and
                              balanced_comparison['passed'] and balanced_pure_scoring['passed'] and
                              (gpu_complete is None or gpu_complete['passed'])}
            result['cases'].append(case)
            print('K=' + str(shot) + ' ' + json.dumps(case), flush=True)
            save(result)
        all_statistics = [case['statistics'] for case in result['cases']]
        all_scores = []
        for case in result['cases']:
            all_scores.extend([case[key] for key in [
                'cached_pipeline_vs_official_cpu', 'same_inputs_scoring_vs_official_cpu',
                'balanced_102_query_cached_pipeline_vs_official_cpu',
                'balanced_102_query_same_inputs_scoring_vs_official_cpu',
                'cached_pipeline_vs_actual_clip_evaluator']])
            gpu = case['full_cuda_statistics_and_actual_evaluator']
            if gpu is not None:
                all_statistics.append(gpu['statistics'])
                all_scores.extend([gpu['scores'], gpu['balanced_102_query_scores']])
        result['maxerrors'] = {
            'image_embedding': image_error['max_abs'], 'text_embedding': text_error['max_abs'],
            **{name: max(stats[name]['max_abs'] for stats in all_statistics)
               for name in ['prototype', 'covariance', 'precision', 'background_mean', 'background_precision']},
            **{name: max(scores[name]['errors']['max_abs'] for scores in all_scores)
               for name in ['cosine', 'md', 'h0']},
        }
        result['prediction_agreement'] = {name: min(scores[name]['prediction_agreement'] for scores in all_scores)
                                          for name in ['cosine', 'md', 'h0']}
        result['source_unchanged_during_verification'] = (
            result['provenance']['core_sha256'] == sha(ROOT/'src/hycal_core.py') and
            result['provenance']['runner_sha256'] == sha(ROOT/'scripts/run_experiments.py'))
        result['passed'] = (result['cache_reencoding']['passed'] and
                            all(case['passed'] for case in result['cases']) and
                            result['source_unchanged_during_verification'])
        result['seconds'] = time.perf_counter() - started
        save(result)
        print(json.dumps({'passed': result['passed'], 'seconds': result['seconds'], 'output': str(OUTPUT)}), flush=True)
        return 0 if result['passed'] else 1
    except Exception:
        result['exception'] = traceback.format_exc()
        result['seconds'] = time.perf_counter() - started
        save(result)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
