from sentence_transformers import SentenceTransformer
modelPath = "models"

model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
model.save(modelPath)
model = SentenceTransformer(modelPath)