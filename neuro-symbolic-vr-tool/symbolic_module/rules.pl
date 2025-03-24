% rules.pl
:- dynamic object/1.

action(create(X)) :- \+ object(X), assertz(object(X)), format('Created ~w~n', [X]).
action(delete(X)) :- object(X), retract(object(X)), format('Deleted ~w~n', [X]).
action(move(X, Pos)) :- object(X), format('Moved ~w to ~w~n', [X, Pos]).
action(_) :- write('Action not recognized or not allowed.'), nl.
