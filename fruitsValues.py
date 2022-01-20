class Apple():
    def __init__(self):
        self.name = "Apple"
        self.kcal = 52
        self.proteins = 0.2
        self.hydrates = 11.4
        self.fat = 0.3
        self.color = (0, 0, 255)
        self.d = 30

    def toString(self):
        return "Kcal: " + str(self.kcal) + "\n" + "Proteins: " + str(self.kcal) + "\n" + "Hydrates: " + str(
            self.kcal) + "\n" + "Fat: " + str(self.kcal)


class Banana():
    def __init__(self):
        self.name = "Banana"
        self.kcal = 89
        self.proteins = 1.3
        self.hydrates = 27
        self.fat = 0.3
        self.color = (0, 255, 255)
        self.d = 60

    def toString(self):
        return "Kcal: " + str(self.kcal) + "\n" + "Proteins: " + str(self.kcal) + "\n" + "Hydrates: " + str(
            self.kcal) + "\n" + "Fat: " + str(self.kcal)


class unDefine():
    def __init__(self):
        self.name = "UnDefined"
        self.kcal = 0
        self.proteins = 0
        self.hydrates = 0
        self.fat = 0
        self.color = (255, 255, 255)

    def toString(self):
        return "Kcal: " + str(self.kcal) + "\n" + "Proteins: " + str(self.kcal) + "\n" + "Hydrates: " + str(
            self.kcal) + "\n" + "Fat: " + str(self.kcal)
