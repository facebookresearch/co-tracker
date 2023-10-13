# coding: utf-8
class SingleImageInferenceInterface:
    def inference(self, image, ann):
        raise NotImplementedError()

    def get_out_meta(self):
        raise NotImplementedError()
