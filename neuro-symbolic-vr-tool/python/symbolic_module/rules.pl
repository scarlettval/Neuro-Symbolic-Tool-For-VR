% ===============================================
% Symbolic Rules for Neuro-Symbolic VR Tool
% ===============================================
loaded_rules :- format('✅ rules.pl successfully loaded~n').
python_alive :- format('🐍 Python connected to correct rules.pl~n').

:- dynamic object/2.

% Initial scene objects (optional)
object(small_red_cube, [size(small), color(red), shape(cube)]).
object(medium_blue_sphere, [size(medium), color(blue), shape(sphere)]).
object(large_green_cylinder, [size(large), color(green), shape(cylinder)]).

% ---------------------------
% Symbolic Actions
% ---------------------------

move_object(Name, Direction) :-
    object(Name, _),
    format('Moving ~w to the ~w~n', [Name, Direction]).

create_object(Name, Properties) :-
    \+ object(Name, _),
    assertz(object(Name, Properties)),
    format('Created ~w with properties ~w~n', [Name, Properties]).

delete_object(Name) :-
    object(Name, _),
    retract(object(Name, _)),
    format('Deleted ~w~n', [Name]).

% ---------------------------
% Helper to build names like small_red_cube
% ---------------------------

make_object_name(Size, Color, Shape, Name) :-
    atomic_list_concat([Size, Color, Shape], '_', Name).

% ---------------------------
% Command Parsing
% ---------------------------

% Interpret: move the SIZE COLOR SHAPE to DIRECTION
interpret(CommandStr, move_object(Name, DirectionAtom)) :-
    split_string(CommandStr, " ", "", Parts),
    Parts = ["move", "the", SizeStr, ColorStr, ShapeStr, "to", DirectionStr],
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    atom_string(DirectionAtom, DirectionStr),
    make_object_name(Size, Color, Shape, Name),
    object(Name, _).


interpret(CommandStr, create_object(Name, [size(Size), color(Color), shape(Shape)])) :-
    split_string(CommandStr, " ", "", Parts),
    Parts = ["create", "the", SizeStr, ColorStr, ShapeStr],
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    make_object_name(Size, Color, Shape, Name),
    \+ object(Name, _).

interpret(CommandStr, delete_object(Name)) :-
    split_string(CommandStr, " ", "", Parts),
    Parts = ["delete", "the", SizeStr, ColorStr, ShapeStr],
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    make_object_name(Size, Color, Shape, Name),
    object(Name, _).
