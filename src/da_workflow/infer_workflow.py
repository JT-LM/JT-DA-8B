from table.POT_reason import POT_Reasoner


class InferWorflow:
    def __init__(self, config):
        self.config = config
        self.reasoner = POT_Reasoner(config)

    def reason_one(self, sample):  # sample的history只保存user和ai的对话，不保存system
        if not sample.history and sample.df is not None:
            init_prompt = self.prompt(sample)
            sample.history.extend(init_prompt)
        new_sample = self.reasoner.reason()

    def reason(self, samples):
        if len(samples) == 0:
            return None
        return [self.reason_one(samples[0])]
    
    def csv_file_read(self,):
        pass