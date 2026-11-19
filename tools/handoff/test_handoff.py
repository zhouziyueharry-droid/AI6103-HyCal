"""Independent, standard-library regression checks for the delivery utilities."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from verify_predictions import prediction_metrics, compare_record, save_outputs, selected_runs
from package_handoff import build_archive, verify_archive


class PredictionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'p.csv'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows):
        with self.path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'true_label', 'pred_label', 'correct'])
            writer.writerows(rows)

    def test_macro_differs_from_top1_and_is_hand_computed(self):
        self.write([['a',0,0,1], ['b',0,0,1], ['c',0,1,0], ['d',1,0,0]])
        metrics, labels = prediction_metrics(self.path, 2, 4)
        self.assertAlmostEqual(metrics['top1'], 50)
        self.assertAlmostEqual(metrics['macro'], 100/3)
        self.assertEqual(metrics['class_counts'], [3,1])
        self.assertEqual(labels, {'a':0,'b':0,'c':0,'d':1})

    def test_invalid_prediction_or_correct_flag_rejected(self):
        for rows in ([['a',0,2,0],['b',1,1,1]], [['a',0,0,0],['b',1,1,1]]):
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    prediction_metrics(self.path,2,2)

    def test_duplicate_or_missing_class_or_wrong_count_rejected(self):
        for rows in ([['a',0,0,1],['a',1,1,1]], [['a',0,0,1],['b',0,0,1]], [['a',0,0,1]]):
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    prediction_metrics(self.path,2,2)

    def test_mismatching_summary_rejected(self):
        actual={'n':2,'macro':50.,'top1':50.,'per_class':[100.,0.],'class_counts':[1,1]}
        compare_record(actual,actual,'valid')
        bad=dict(actual,macro=51.)
        with self.assertRaises(ValueError):
            compare_record(actual,bad,'corrupt')

    def test_joint_equal_domain_metric_is_not_class_macro(self):
        self.write([[str(i),i,i if i<102 else 0,int(i<102)] for i in range(149)])
        actual,_=prediction_metrics(self.path,149,149)
        self.assertEqual(actual['equal_domain_macro'],50.)
        self.assertAlmostEqual(actual['macro'],100*102/149)

    def test_missing_duplicate_seeds_and_changed_config_rejected(self):
        base=self.root/'results/part_a_round2'
        (base/'propagation').mkdir(parents=True)
        (base/'toy/test').mkdir(parents=True)
        (base/'selection.json').write_text(json.dumps({'selected':[{'config':{'family':'H0'}}]}))
        (base/'propagation/selection.json').write_text(json.dumps({'selected':[]}))
        row={'seed':0,'dataset':'toy','phase':'test','shot':5,'tag':'s0_H0','config':{'family':'H0'}}
        bad_config=dict(row,seed=1,config={'family':'H0','extra':True})
        for rows in ([row],[row,row],[row,bad_config]):
            with self.subTest(rows=rows):
                (base/'toy/test/summary.json').write_text(json.dumps(rows))
                with patch('verify_predictions.ENDPOINTS',(('toy','test',5,2,4,(0,1)),)), patch('verify_predictions.FAMILIES',('H0',)):
                    with self.assertRaises(ValueError):
                        list(selected_runs(self.root))

    def test_output_is_new_and_protects_research_trees(self):
        payload={'passed':True,'aggregates':[]}
        out=self.root/'verification_runs'/'first'
        save_outputs(payload,out,self.root)
        self.assertTrue((out/'verification.json').exists())
        with self.assertRaises(FileExistsError):
            save_outputs(payload,out,self.root)
        with self.assertRaises(ValueError):
            save_outputs(payload,self.root/'results'/'new',self.root)

    def test_archive_integrity_membership_and_no_overwrite(self):
        import hashlib
        source=self.root/'example.txt'
        source.write_bytes(b'original evidence')
        manifest={'files':[{'archive_path':'example.txt','source_relative_path':'example.txt',
                           'size_bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]}
        archive=self.root/'delivery.zip'
        build_archive(self.root,archive,manifest)
        self.assertTrue(verify_archive(archive)['passed'])
        with self.assertRaises(FileExistsError):
            build_archive(self.root,archive,manifest)
        with zipfile.ZipFile(archive,'a') as z:
            z.writestr('unlisted.txt','unexpected')
        with self.assertRaises(ValueError):
            verify_archive(archive)

    def test_archive_rejects_protected_research_output(self):
        for relative in ('results/new.zip','src/new.zip','docs/new.zip'):
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    build_archive(self.root,self.root/relative,{'files':[]})
                self.assertFalse((self.root/relative).exists())


if __name__ == '__main__':
    unittest.main()
