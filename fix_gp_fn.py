import os
import re
import transliterate


def _fix_name(name) -> str:
    if re.search(r"[а-яА-Я]", name):
        name = transliterate.translit(name, "ru", reversed=True)
    name = re.sub(r"\s+", "_", name)  # replace whitespaces
    name = re.sub(r"\'", "-", name)  # replace apostrophes
    return name


def transliterate_file_names(directory):
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        fixed_name = _fix_name(name)
        if fixed_name != name:
            fixed_path = os.path.join(directory, fixed_name)
            print(f"Will rename {path} to {fixed_path}")
            if os.path.exists(fixed_path):
                print(f"Path {fixed_path} already exists, just remove original file?")
                if input() == "y":
                    os.remove(path)
                continue
            os.rename(path, fixed_path)
            path = fixed_path
        if os.path.isdir(path):
            transliterate_file_names(path)


root_directory = "/Users/vlad/[archive] Google Drive/PlayMusic/tabs"
transliterate_file_names(root_directory)
