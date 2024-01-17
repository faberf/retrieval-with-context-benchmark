from transformers import CLIPModel
from transformers import AutoTokenizer, AutoProcessor

class ClipVitLargePatch14():

    def load_model(self):

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    def text_embedding(self, text, config={}):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")

        outputs = self.model.get_text_features(**inputs, **config)
        return outputs.tolist()

    def image_embedding(self, image, config={}):
        inputs = self.processor(images=list(map(lambda x: x.to_numpy(), image)), return_tensors="pt")

        outputs = self.model.get_image_features(**inputs, **config)
        return outputs.tolist()


    def zero_shot_image_classification(self, image, classes, config={}):
        inputs = self.processor(text=classes, images=list(map(lambda x: x.to_numpy(), image)), return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        return probs.tolist()
    
    def zero_shot_image_retrieval(self, image, text, config={}):
        inputs = self.processor(text=text, images=list(map(lambda x: x.to_numpy(), image)), return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs.tolist()