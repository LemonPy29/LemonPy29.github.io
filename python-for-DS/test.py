def reader(sep):
    def read(path):
        return sep + path
    return read

r = reader(';')
res = r('mydir')
print(res)
