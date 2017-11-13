import glob
import imageio

type = 'GuidedBackprop'
root_dir = './results/10112017/'
path = root_dir + type
images = []
for file in glob.glob("{}/*/*.png".format(path)):
    print(file)
    images.append(imageio.imread(file))
imageio.mimsave('{}/{}.gif'.format(root_dir), images)









