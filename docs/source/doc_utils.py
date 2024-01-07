import os
import glob


def generate_examples_gallery_rst(
    title: str = "Examples",
    filename: str = "examples_gallery",
    foldername: str = "examples",
    reversed: bool = False,
):
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    gallery_rst_path = os.path.join(folder_path, filename + ".rst")
    gallery_folder_path = os.path.join(folder_path, foldername)
    notebook_list = glob.glob(os.path.join(gallery_folder_path, "*.ipynb"))
    notebook_names = [os.path.basename(file_path) for file_path in notebook_list]

    with open(gallery_rst_path, "w") as f:
        f.write(title + "\n")
        f.write(len(title) * "-" + "\n")
        
        f.write("\n")
        
        f.write(r".. nbgallery::" + "\n")
        f.write(r"    :maxdepth: 1" + "\n")
        
        if reversed:
            f.write(r"    :reversed:" + "\n")
            
        f.write("\n")

        for nb in notebook_names:
            f.write(r"    {foldername}/".format(foldername=foldername) + nb + "\n")


def generate_user_guide_rst(
    title: str = "User Guide",
    filename: str = "user_guide",
    foldername: str = "notebooks",
    reversed: bool = False,
    description: str = None,
):
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    gallery_rst_path = os.path.join(folder_path, filename + ".rst")
    gallery_folder_path = os.path.join(folder_path, foldername)
    notebook_list = glob.glob(os.path.join(gallery_folder_path, "*.ipynb"))
    notebook_names = [os.path.basename(file_path) for file_path in notebook_list]

    with open(gallery_rst_path, "w") as f:
        f.write(title + "\n")
        f.write(len(title) * "-" + "\n")
        
        if description:
            f.write(description + "\n")
        
        f.write("\n")
        
        f.write(r".. toctree::" + "\n")
        f.write(r"    :maxdepth: 2" + "\n")
        if reversed:
            f.write(r"    :reversed:" + "\n")
        f.write("\n")

        for nb in notebook_names:
            f.write(r"    {foldername}/".format(foldername=foldername) + nb + "\n")
