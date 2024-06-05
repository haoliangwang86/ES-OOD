IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
BATCH_SIZE = 128
OOD_SAMPLE_SIZE = 10000
IND_SAMPLE_SIZE = 50000

VGG16_LAYERS = range(13)
RESNET34_LAYERS = range(34)
DENSENET100_LAYERS = range(51)

OOD_LIST = ["lsun", "tiny", "svhn", "dtd", "pure_color"]

vgg_selected_layers = list(range(13))
resnet_selected_layers = list(range(9)) + list(range(10, 18)) + list(range(19, 31)) + list(range(32, 37))
densenetnet_selected_layers = list(range(1, 32, 2)) + [33] + list(range(34, 65, 2)) + [66] + list(range(67, 98, 2)) + [99]
