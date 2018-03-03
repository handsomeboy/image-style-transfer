import os

from style_transfer.transfer import Transfer

WIDTH = 100
HEIGHT = 100

DATA_INPUT = "style_transfer/data/input/"
DATA_OUTPUT = "style_transfer/data/output/"

style_path = os.path.join(DATA_INPUT,"style/vangogh.jpg")
content_path = os.path.join(DATA_INPUT,"content/baker.jpg")

synthetic_name = os.path.splitext(style_path)[0].split("/")[-1]
synthetic_name += "_on_"
synthetic_name += os.path.splitext(content_path)[0].split("/")[-1]

transfer = Transfer(style_path, content_path, WIDTH, HEIGHT)


full = transfer.open_image(os.path.join(DATA_INPUT , "content/baker.jpg"))
half = transfer.open_image(os.path.join(DATA_INPUT , "content/half_baker.jpg"))
quarter = transfer.open_image(os.path.join(DATA_INPUT , "content/quarter_baker.jpg"))

full_loss = transfer.get_content_loss(full)
half_loss = transfer.get_content_loss(half)
quarter_loss = transfer.get_content_loss(quarter)
syn_loss = transfer.get_content_loss(transfer.synthetic)

print("Full =" + str(full_loss / syn_loss))
print("Half =" + str(half_loss / syn_loss))
print("Quarter =" + str(quarter_loss / syn_loss))
print("Synth = " + str(syn_loss / syn_loss))
