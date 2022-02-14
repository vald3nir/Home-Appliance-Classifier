from subprocess import call


def write_array(file, lines):
    f = open(file, "w")
    for line in lines:
        f.write(line)
    f.close()


def load_libraries(libraries):
    call("pip install " + ' '.join(libraries), shell=True)
    # call("pip freeze > requirements.txt", shell=True)
    write_array("requirements.txt", [l + "\n" for l in libraries])


if __name__ == '__main__':
    load_libraries(libraries=[
        "matplotlib", "numpy", "pandas",
        "scikit-learn", "jupyterlab", "notebook",
    ])
