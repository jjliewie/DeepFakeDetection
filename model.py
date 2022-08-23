class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		model_ft = models.resnet18(pretrained=True)
		self.encoder = torch.nn.Sequential(*list(model_ft.children())[::-1])
		self.decoder = DenseNetDecoder()
		self.classifer = nn.Linear(512, 2)
	
	def forward(self, face):
		latent_var = self.encoder(face)
		reconstructed_face = self.decoder(latent_var)
		deepfake_feature = latent_var[:, 0:128, :, :]
		pred = self.classifier(deepfake_feature)
		return reconstructed_face, pred
