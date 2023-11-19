import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from woe_binning import Binning
from sklearn.metrics import roc_auc_score

class ScorecardPipeline(BaseEstimator, TransformerMixin):

    def __init__(self, min_bin_pct=0.025, y_threshold=10, p_threshold=0.35, max_pct_missing_threshold=0.8, test_set_size=0.2, seed=123456789, min_iv=0, max_iv=100, max_features=20, correlation_cutoff=0.4, cv_folds=5, ref_odds=15, ref_score=660, pdo=20):
        self.min_bin_pct = min_bin_pct
        self.y_threshold = y_threshold
        self.p_threshold = p_threshold
        self.max_pct_missing_threshold = max_pct_missing_threshold
        
        self.test_set_size = test_set_size
        self.seed = seed
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.max_features = max_features
        self.correlation_cutoff = correlation_cutoff
        self.cv_folds = cv_folds
        self.ref_odds = ref_odds
        self.ref_score = ref_score
        self.pdo = pdo

        self.train_index = object
        self.test_index = object
        self.woe_tables = pd.DataFrame()
        self.scorecard = pd.DataFrame()
        self.performance = pd.DataFrame()
        self.bin_obj = object
        self.model = object

    def lasso_binary_feature_selection(self, x_train, y_train):

        cv = StratifiedKFold(n_splits = self.cv_folds, random_state = self.seed, shuffle = True)
        model = LassoCV(cv = cv, random_state = self.seed, max_iter = 100000, n_jobs=-1)
        model.fit(x_train, y_train)
        
        coef = pd.Series(model.coef_, index = x_train.columns)

        coef = pd.DataFrame(coef)
        coef['feature'] = coef.index
        coef.columns = ['coef','feature']

        coef = coef.sort_values('coef', ascending=False)

        selected = list(coef.loc[abs(coef['coef']) != 0]['feature'])
        return selected

    def get_correlated_features(self, data, woe_df):

        iv_summary = woe_df[['feature','total_iv']].copy().drop_duplicates()

        cm = data.copy().corr(numeric_only=True)
        cm = cm.abs() > self.correlation_cutoff

        to_remove = []
        for feature in cm.columns: 

            iv_tmp = iv_summary.loc[iv_summary['feature'] == feature]['total_iv'].item()
            corr_feats = cm.loc[cm.columns == feature]
            corr_feats = corr_feats.transpose()
            corr_feats = corr_feats.loc[corr_feats[feature] == True]
            corr_feats['feature'] = corr_feats.index
            corr_feats = corr_feats.merge(iv_summary, how = 'left', on = 'feature')
            corr_feats = corr_feats.loc[corr_feats['total_iv'] > iv_tmp]

            if len(corr_feats) > 0:
                to_remove.append(feature)
                
        return to_remove

    def tune_and_train_logistic_regression(self, train, features, target):

        lr = LogisticRegression(class_weight = 'balanced', random_state = self.seed)

        param_grid = [
            {'penalty' : ['l1', 'l2'],
            'solver':['liblinear'],
            'C' : np.logspace(-1, 1, 50)},
            {'penalty' : ['l1', 'l2', ],
            'solver':['saga'],
            'C' : np.logspace(-1, 1, 50)}
        ]

        random_grid = RandomizedSearchCV(
            estimator = lr,
            param_distributions = param_grid,
            scoring = 'roc_auc',
            n_jobs = -1,
            cv = self.cv_folds,
            refit = True,
            random_state=self.seed,
            return_train_score = True)

        random_grid.fit(train[features], train[target])
        best = random_grid.best_params_
        
        lr = LogisticRegression(max_iter=100000, penalty=best['penalty'], C=best['C'], solver=best['solver'], n_jobs=-1, class_weight = 'balanced', random_state = self.seed)
        lr.fit(train[features], train[target])

        cv = StratifiedKFold(n_splits=self.cv_folds, random_state=self.seed, shuffle=True)
        scores = cross_val_score(lr, train[features], train[target], cv=cv, scoring="roc_auc", n_jobs=-1)
        AUROC = scores.mean()
        cv_auc = AUROC
        cv_gini = AUROC * 2 - 1
        
        return lr, cv_auc, cv_gini

    def create_scorecard(self, model, final_features, iv_df):

        def calculate_probability(score):
            return 1 / (1 + np.exp(-score))

        factor = self.pdo / np.log(2)
        offset = self.ref_score - factor * np.log(self.ref_odds)
        intercept = model.intercept_
        nr_features = len(final_features)

        scorecard = pd.DataFrame({'feature':final_features})
        scorecard['coefficient'] = model.coef_[0]
        scorecard = scorecard.merge(iv_df[['feature','from_value_incl','to_value_excl','bin','woe']], how='left', on='feature')
        scorecard['score'] = (scorecard['coefficient']*(scorecard['woe']) + intercept/nr_features) * factor+offset/nr_features
        scorecard = scorecard.loc[scorecard['feature'].isin(final_features)]
        scorecard['from_value_incl'] = np.where(scorecard['bin'] == 'Missing', 'Missing', scorecard['from_value_incl'])
        scorecard['to_value_excl'] = np.where(scorecard['bin'] == 'Missing', 'Missing', scorecard['to_value_excl'])

        return scorecard

    def fit(self, data, y, id_col=None):

        self.y = y
        self.id_col = id_col
        if self.id_col is None:
            self.id_col = []

        if self.test_set_size > 0:
            print('Partitioning data...')
            x_train, x_test, y_train, y_test = train_test_split(data[data.columns.drop(self.y)], data[self.y], test_size=self.test_set_size, random_state=self.seed)
            self.train_index = x_train.index
            self.test_index = x_test.index
        else:
            x_train = data[data.columns.drop(self.y)]
            y_train = data[self.y]
            x_test = None
            y_test = None
            self.train_index = x_train.index
            self.test_index = None
        
        print('Calculating optimal bins...')
        self.bin_obj = Binning(min_bin_pct=self.min_bin_pct, y_threshold=self.y_threshold, p_threshold=self.p_threshold, max_pct_missing_threshold=self.max_pct_missing_threshold, show_progress=True)
        self.bin_obj.fit(data=pd.concat([x_train, y_train], axis=1), y=self.y, id_col=self.id_col)
        woe_df = self.bin_obj.woe_tables.copy()
        woe_df = woe_df.loc[(woe_df['total_iv'] >= self.min_iv) & (woe_df['total_iv'] <= self.max_iv)]
        self.woe_tables = woe_df
        features = woe_df['feature'].unique()
        train_df = self.bin_obj.transform(pd.concat([x_train, y_train], axis=1), woe_df)
 
        if self.test_set_size > 0:
            test_df = self.bin_obj.transform(pd.concat([x_test, y_test], axis=1), woe_df)

        print('Reducing features...')
        features = self.lasso_binary_feature_selection(train_df[features], train_df[self.y])

        features.append(self.y)
        if self.id_col != []:
            features.append(self.id_col)

        train_df = train_df[features]
        if self.test_set_size > 0:
            test_df = test_df[features]
        
        features.remove(self.y)
        if self.id_col != []:
            features.remove(self.id_col)

        to_remove = self.get_correlated_features(train_df[features], woe_df)
        train_df.drop(to_remove, axis=1, inplace=True)
        if self.test_set_size > 0:
            test_df.drop(to_remove, axis=1, inplace=True)

        final_features = train_df.columns.drop(self.y)
        if self.id_col != []:
            final_features = final_features.drop(self.id_col)

        imp_df = woe_df[['feature','total_iv']].drop_duplicates().sort_values('total_iv', ascending=False)
        final_features = imp_df.loc[imp_df['feature'].isin(final_features)]['feature'].unique()[0:self.max_features]
        print('Training model...')
        lr, cv_auc, cv_gini = self.tune_and_train_logistic_regression(train_df, final_features, self.y)
        self.model = lr

        p_train = lr.predict_proba(train_df[final_features])[:,1]
        train_auc = roc_auc_score(y_train, p_train)
        train_gini = train_auc * 2 - 1
        if self.test_set_size > 0:
            p_test = lr.predict_proba(test_df[final_features])[:,1]
            test_auc = roc_auc_score(y_test, p_test)
            test_gini = test_auc * 2 - 1
        else:
            test_auc = None
            test_gini = None

        self.performance = pd.DataFrame({
            'train_gini':train_gini,
            'cv_gini':cv_gini,
            'test_gini':test_gini,
            'train_auc':train_auc,
            'cv_auc':cv_auc,
            'test_auc':test_auc
        }, index=[0])

        print('Creating scorecard...')
        self.scorecard = self.create_scorecard(lr, final_features, woe_df)

    def transform(self, df, scorecard, include_only_final_score=False):
        out_df = self.bin_obj.transform(df.copy(), self.woe_tables)

        out_df['probability'] = self.model.predict_proba(out_df[self.model.feature_names_in_])[:,1]
        for feature in scorecard['feature'].unique():

            out_df[feature + '_score_'] = None
            tmp_score = scorecard.loc[scorecard['feature'] == feature]

            for row in range(tmp_score.shape[0]):
                tmp = tmp_score.iloc[row]
                score = tmp['score']
                woe = tmp['woe']

                out_df[feature + '_score_'] = np.where(out_df[feature] == woe, score, out_df[feature + '_score_'])

        include = out_df.columns[out_df.columns.str.contains('_score_')]
        out_df['_model_score_'] = out_df[include].sum(axis=1)
        out_df['target'] = df[self.y]
        
        out_df = out_df[list(include) + list(['_model_score_']) + list(['probability']) + list(['target'])]
        if include_only_final_score:
            return out_df['_model_score']
        else: 
            return out_df