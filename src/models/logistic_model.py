from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

class LogisticEmotionClassifier:
    def __init__(self):
        self.model = None
        self.best_params = None
    
    def build_model(self):
        """Build pipeline with TF-IDF and Logistic Regression"""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        return pipeline
    
    def train(self, X_train, y_train, tune_hyperparams=True):
        """Train model with optional hyperparameter tuning"""
        if tune_hyperparams:
            param_grid = {
                'tfidf__max_features': [3000, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'clf__C': [0.1, 1.0, 10.0],
                'clf__penalty': ['l2']
            }
            
            pipeline = self.build_model()
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, 
                                      scoring='f1_macro', n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            self.model = self.build_model()
            self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        self.model = joblib.load(filepath)