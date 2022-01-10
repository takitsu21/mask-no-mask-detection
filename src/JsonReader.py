import json
from PIL import Image
import os

import tensorflow as tf
from pathlib import Path
import re


class Spec(object):
    def __init__(self,
                 categorie: str,
                 begin: tuple[int, int],
                 dest: tuple[int, int],
                 imgSize: tuple[int, int]) -> None:
        self.categorie = categorie
        self.begin = begin
        self.dest = dest
        self.imgSize = imgSize

    def __repr__(self) -> str:
        return f"{self.categorie} {self.begin} {self.dest} {self.imgSize}"


class MyImage(object):
    def __init__(self,
                 absolutePath: str,
                 path: str,
                 ) -> None:
        self.absolutePath = absolutePath
        self.name = path.split("\\")[-1].split(".")[0]
        self.path = path
        self.specs: list[Spec] = []

    def addSpec(self, spec: Spec) -> None:
        self.specs.append(spec)

    def __repr__(self) -> str:
        return f"<{self.name} {self.categorie} {self.begin} {self.dest} {self.imgSize} {self.path}>"


class JsonReader(object):

    def __init__(self, path: str) -> None:
        self.path = path
        self.images: list[MyImage] = []

    def readJson(self):
        with open(self.path, 'r') as f:
            return json.load(f)

    def parseJson(self):
        jsonData: dict = self.readJson()

        for absolutePath, v in jsonData.items():

            annotations = v["annotations"]
            if len(annotations):
                imageSpec = MyImage(
                    absolutePath,
                    annotations[0]["path"]
                )
                for annotation in annotations:
                    coordsBegin = annotation['coords']['beginX'], annotation['coords']['beginY']
                    coordsEnd = annotation['coords']['destinationX'], annotation['coords']['destinationY']
                    imgSize = annotation['img-size']['width'], annotation['img-size']['height']
                    imageSpec.addSpec(Spec(
                        annotation['categorie'],
                        coordsBegin,
                        coordsEnd,
                        imgSize
                    ))

                self.images.append(imageSpec)

        self.convertImages()
        return self.images

    def convertImages(self):
        print("Start converting images")
        if not os.path.exists(f"{os.getcwd()}/converted-images"):
            os.makedirs(f"{os.getcwd()}/converted-images")

        print(os.getcwd())
        for img in self.images:
            myPath = Path(img.path)

            myPath = re.sub(r'\\', '/', str(myPath))
            image = Image.open(myPath)
            for spec in img.specs:

                bX, bY = spec.begin
                dX, dY = spec.dest

                cropped = image.crop((bX, bY, dX, dY))
                imgName = f"{img.name}-bb-{bX}x{bY}-{dX-bX}-{dY-bY}.png"
                pathToSave = Path(f"{os.getcwd()}/converted-images/{imgName}")
                cropped.save(pathToSave)

                print(f"{spec.categorie} Annotated {imgName} has been converted.")
        return self.images

    def getImages(self) -> list[MyImage]:
        return self.images


if __name__ == "__main__":
    reader = JsonReader("img/tests/tests-annotation2.json")
    reader.parseJson()
