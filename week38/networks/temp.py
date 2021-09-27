class Grandparent:
    def __init__(self):
        self.child = Parent(self)
        self.num = self.child.num

    def update_num(self):
        self.num = self.child.num

class Parent(Grandparent):
    def __init__(self, parent):
        self.parent = parent
        self.child = Child(self)
        self.num = self.child.num

    def update_num(self):
        self.num = self.child.num

class Child(Grandparent):
    def __init__(self, parent):
        self.parent = parent
        self.num = 3.1415

    def change_num(self, num):
        self.num = num
        self.parent.update_num()
        self.parent.parent.update_num()

p1 = Grandparent()
print(p1.num)
p1.child.child.num = 2
print(p1.num)
p1.child.child.change_num(2)
print(p1.num)


