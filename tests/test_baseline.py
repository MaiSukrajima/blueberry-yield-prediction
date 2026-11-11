"""Regression checks for fold isolation, saved scores, and prediction interfaces."""
from __future__ import annotations
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loader import EXPECTED_FEATURES, iter_cv
from src.models.predict_baseline import feature_frame, predict
from src.models.train_baseline import ROOT, build_models, load_inputs, metrics, source_info


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'configs/baseline.json').read_text(encoding='utf-8'))
        cls.df,cls.folds,cls.cv=load_inputs(ROOT,cls.config)
        cls.out=ROOT/'reports/baseline_models'
        cls.manifest=json.loads((cls.out/'metrics.json').read_text(encoding='utf-8'))
        cls.oof=pd.read_csv(cls.out/'oof_predictions.csv')
        cls.per=pd.read_csv(cls.out/'fold_metrics.csv')

    def test_frozen_partition_and_exact_oof_coverage(self):
        self.assertEqual(len(self.oof),621)
        self.assertTrue(self.oof['Row#'].is_unique)
        self.assertEqual(set(self.oof['Row#']),set(self.df.iloc[self.cv]['Row#']))
        hold=set(self.df.loc[self.folds['fold_id'].eq(-1),'Row#'])
        self.assertTrue(hold.isdisjoint(set(self.oof['Row#'])))
        seen=[]
        for tr,va in iter_cv(self.folds):
            self.assertTrue(set(tr).isdisjoint(set(va)))
            self.assertTrue((self.folds.iloc[tr]['fold_id']>=0).all())
            self.assertTrue((self.folds.iloc[va]['fold_id']>=0).all())
            seen.extend(va)
        self.assertEqual(sorted(seen),sorted(self.cv.tolist()))

    def test_aggregate_metrics_recalculate_from_oof(self):
        for result in self.manifest['results']:
            got=metrics(self.oof['yield'],self.oof[result['model']])
            for metric,value in got.items():
                self.assertAlmostEqual(value,result['oof_'+metric],places=6)
            per=self.per.loc[self.per.model.eq(result['model'])]
            self.assertAlmostEqual(per.valid_mae.std(ddof=1),result['fold_mae_std'],places=8)
            weighted=np.average(per.valid_mae,weights=per.valid_n)
            self.assertAlmostEqual(weighted,result['oof_mae'],places=8)

    def test_saved_fold_models_reproduce_oof_and_fold_scalers(self):
        X=self.df.iloc[self.cv][EXPECTED_FEATURES].reset_index(drop=True)
        for key in self.config['models']:
            for fold in range(5):
                mask=self.oof.fold_id.eq(fold).to_numpy()
                model=joblib.load(self.out/'models'/f'{key}_fold{fold}.joblib')
                self.assertEqual(list(model.feature_names_in_),EXPECTED_FEATURES)
                np.testing.assert_allclose(model.predict(X.loc[mask]),self.oof.loc[mask,key],rtol=1e-10,atol=1e-7)
                if 'scaler' in model.named_steps:
                    scaler=model.named_steps['scaler']
                    self.assertEqual(scaler.n_samples_seen_,int((~mask).sum()))
                    np.testing.assert_allclose(scaler.mean_,X.loc[~mask].mean().to_numpy(),rtol=1e-12)

    def test_reference_scores_reproduced(self):
        for result in self.manifest['reference_reproduction'].values():
            self.assertLess(result['absolute_delta'],1e-6)

    def test_cv621_checkpoints_have_no_holdout(self):
        self.assertFalse(self.manifest['holdout_used'])
        self.assertIsNone(self.manifest['holdout_mae'])
        self.assertFalse(self.manifest['final_all777_model'])
        for key,checkpoint in self.manifest['checkpoints'].items():
            self.assertEqual(checkpoint['trained_rows'],621)
            self.assertFalse(checkpoint['includes_holdout'])
            path=self.out/checkpoint['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint['sha256'])
            model=joblib.load(path)
            self.assertEqual(model.n_features_in_,16)
            if 'scaler' in model.named_steps:
                self.assertEqual(model.named_steps['scaler'].n_samples_seen_,621)

    def test_inference_column_order_and_row_ids(self):
        example=pd.read_csv(self.out/'inference_example.csv')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            frame=example[example.columns[::-1]]
            frame.to_csv(path/'input.csv',index=False)
            result=predict(self.out/'models/linear_regression_cv621.joblib',path/'input.csv',path/'output.csv')
            self.assertEqual(result['Row#'].tolist(),example['Row#'].tolist())
            model=joblib.load(self.out/'models/linear_regression_cv621.joblib')
            np.testing.assert_allclose(result['predicted_yield'],model.predict(example[EXPECTED_FEATURES]))

    def test_inference_rejects_bad_schema_and_values(self):
        frame=self.df.iloc[self.cv[:4]][EXPECTED_FEATURES].copy()
        for bad in [frame.drop(columns=EXPECTED_FEATURES[0]),frame.assign(yield_=1),
                    frame.assign(**{'yield':1}),frame.assign(fruitset=np.nan),frame.iloc[:0]]:
            with self.assertRaises(ValueError): feature_frame(bad)

    def test_fingerprint_guard_rejects_changed_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'data/processed').mkdir(parents=True)
            raw=(ROOT/'data/processed/folds.csv').read_bytes()
            (root/'data/processed/folds.csv').write_bytes(raw+b'\n')
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                load_inputs(root,self.config)

    def test_models_and_source_match_manifest(self):
        self.assertEqual(set(build_models(self.config)),set(self.config['models']))
        self.assertNotIn('Row#',self.manifest['feature_columns'])
        self.assertNotIn('yield',self.manifest['feature_columns'])
        self.assertEqual(source_info(ROOT)['source_sha256'],self.manifest['source']['source_sha256'])

    def test_git_tracks_source_and_ignores_generated_data(self):
        if not (ROOT/'.git').exists():
            self.skipTest('Source ZIP contains commit identity but no .git directory')
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        for path in ['src/data/loader.py','src/models/train_baseline.py','src/models/predict_baseline.py']:
            self.assertIn(path,tracked)
        # The team repository keeps data/README.md and .gitkeep; only data files
        # and fitted weights are excluded, not the directory's documentation.
        self.assertFalse(any(
            (p.startswith('data/') and Path(p).suffix.lower() in {'.csv','.xlsx','.zip'})
            or p.endswith('.joblib') for p in tracked))


if __name__=='__main__': unittest.main(verbosity=2)
