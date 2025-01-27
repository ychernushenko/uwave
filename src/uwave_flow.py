from metaflow import FlowSpec, step


class UwaveFlow(FlowSpec):

    @step
    def start(self):
        import pandas as pd

        # Load data
        data = pd.read_parquet("data/gesture_data.parquet")

        self.X = data.drop(["gesture", "user"], axis=1)  # Features
        self.y = data["gesture"]  # Target

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        self.X = self.X.replace([np.inf, -np.inf], np.nan)  # Replace inf/-inf with NaN
        self.X = self.X.fillna(self.X.mean())  # Replace NaN with column means

        X_scaled = StandardScaler().fit_transform(self.X)

        # Split Data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        self.next(self.train_model)

    @step
    def train_model(self):

        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        from sklearn.metrics import classification_report, confusion_matrix

        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

        self.next(self.end)

    @step
    def end(self):
        print("Model training finished!")


if __name__ == "__main__":
    UwaveFlow()
