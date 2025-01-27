from metaflow import FlowSpec, conda_base, step


@conda_base(python="3.10", libraries={"scikit-learn": "1.6.1"})
class UwaveFlow(FlowSpec):

    @step
    def start(self):
        import data_load

        X, y = data_load.load_uwave_dataset()
        self.next(self.train_lg)

    def train_lg(self):
        from sklearn.cluster import KMeans

        with profile("k-means"):
            kmeans = KMeans(n_clusters=10, verbose=1, n_init=1)
            kmeans.fit(self.mtx)
        self.clusters = kmeans.labels_
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    UwaveFlow()
