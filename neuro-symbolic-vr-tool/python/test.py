from pyswip import Prolog

prolog = Prolog()
print(list(prolog.query("current_prolog_flag(version, X)")))
