import ast

def analyze_code(code):
    """

    """
    try:
        # Python kodunu parse ediyoruz
        tree = ast.parse(code)
        # Hata yoksa boş liste döndürüyoruz
        return []
    except SyntaxError as e:
        # Syntax hatasını yakalayıp liste olarak döndürüyoruz
        return [str(e)]
