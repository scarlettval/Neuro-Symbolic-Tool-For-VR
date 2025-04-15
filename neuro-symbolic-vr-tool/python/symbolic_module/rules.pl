% =============================
% Dynamic Symbolic Rules
% =============================

:- dynamic object/2.

% Example object knowledge base
object(small_red_cube, [size(small), color(red), shape(cube)]).

% ----------------------------
% Action Definitions
% ----------------------------

move_object(Name, Direction) :-
    object(Name, _),
    format('Moving ~w to ~w~n', [Name, Direction]).

create_object(Name, [size(Size), color(Color), shape(Shape)]) :-
    \+ object(Name, _),
    assertz(object(Name, [size(Size), color(Color), shape(Shape)])),
    format('Created ~w~n', [Name]).

delete_object(Name) :-
    object(Name, _),
    retract(object(Name, _)),
    format('Deleted ~w~n', [Name]).

% ----------------------------
% Object Name Builder
% ----------------------------

make_object_name(Size, Color, Shape, Name) :-
    atomic_list_concat([Size, Color, Shape], '_', Name).

% ----------------------------
% interpret/2 Rules
% ----------------------------

% move the small red cube to left
interpret(CommandStr, move_object(Name, Direction)) :-
    split_string(CommandStr, " ", "", Words),
    Words = ["move", "the", SizeStr, ColorStr, ShapeStr, "to", DirectionStr],
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    atom_string(Direction, DirectionStr),
    make_object_name(Size, Color, Shape, Name),
    object(Name, _).

% create the small red cube
interpret(CommandStr, create_object(Name, [size(Size), color(Color), shape(Shape)])) :-
    split_string(CommandStr, " ", "", Words),
    Words = ["create", "the", SizeStr, ColorStr, ShapeStr],
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    make_object_name(Size, Color, Shape, Name),
    \+ object(Name, _).

% delete the small red cube
interpret(CommandStr, delete_object(Name)) :-
    split_string(CommandStr, " ", "", Words),
    Words = ["delete", "the", SizeStr, ColorStr, ShapeStr],
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    make_object_name(Size, Color, Shape, Name),
    object(Name, _).
