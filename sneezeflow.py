from metaflow import FlowSpec, step
from utils import clip_data, get_data
import ray

class SneezeFlow(FlowSpec):
    
    @step
    def start(self):
        # which datapoints (i.e. sneezes) we want to include
        self.files = [11, 12, 13]
        self.next(self.clean_data)

    @step
    def clean_data(self):
        # process raw data from motion capture
        futures = []

        for idx in self.files:
            futures.append(clip_data.remote(idx))
        
        ray.get(futures)
        self.next(self.load_data)
        
    @step
    def load_data(self):
        # extract featueres and load data into dataloaders for training
        self.trainDL, self.testDL = get_data(self.files)
        print(f"training set has {len(self.trainDL)} batches")
        print(f"test set has {len(self.testDL)} batches")
        self.next(self.end)

    @step
    def end(self):
        print("All tasks completed.")

if __name__ == "__main__":
    ray.init()
    SneezeFlow()
    ray.shutdown()