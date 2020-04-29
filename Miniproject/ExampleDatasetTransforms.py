from DataLoader import RetinaDataset
#Show cropping and rotation with low probability
test = RetinaDataset(file_path="/data/targets", transforms=[Rotate(p=0.5), RandomCrop(p=0.5, height=300, width=300)])
sample = test[0]
print(sample['image'])
plt.imshow(sample['image'], cmap='gray')
plt.show()
