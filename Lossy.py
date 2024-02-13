import os
from PIL import Image
from zipfile import ZipFile


data_path = "img_data/"
result_path = "img_result/"
file_type = ".png"
img_1_name = "absol"
img_2_name = "mareep"


def img_compression(img_name, quality):
    try:
        img_path = data_path + img_name + file_type
        img = Image.open(img_path)
        img.show("Eva what happened?")
        comp_img_path = result_path + "compressed_" + img_name + file_type
        img.save(comp_img_path, "PNG", quality=quality)
        comp_img = Image.open(comp_img_path)
        comp_img.show("Compressed")
        print(f"Original size: {os.path.getsize(img_path)}")
        print(f"Compressed size: {os.path.getsize(comp_img_path)}")
    except Exception as e:
        print(e)
    return


def zip_files(files_to_zip):
    with ZipFile('zipped_files.zip', 'w') as zipper:
        # writing each file one by one
        for file in files_to_zip:
            zipper.write(file)


def main():
    img_compression(img_1_name, quality=60)


if __name__ == '__main__':
    main()

