%% rules.pl
%% Full symbolic interpretation rules for Neuro-Symbolic VR

:- module(rules, [
    load_scene/1,
    interpret/2,
    add_object/5,
    delete_object/1,
    move_object/4,
    list_objects/0
]).

:- use_module(library(http/json)).
:- dynamic object/6.
:- discontiguous parse_command/2.

%% === Load Scene from JSON ===
load_scene(File) :-
    retractall(object(_,_,_,_,_,_)),
    open(File, read, In),
    json_read_dict(In, Objects),
    close(In),
    maplist(assert_object, Objects).

assert_object(Obj) :-
    string_lower(Obj.get(label), LabelStr),
    atomic_list_concat(Words, ' ', LabelStr),
    atomic_list_concat(Words, '_', Label),
    Color     = Obj.get(color),
    Shape     = Obj.get(shape),
    Material  = Obj.get(material, default),
    Size      = Obj.get(size),
    Position  = Obj.get(position),
    assertz(object(Label, Color, Shape, Material, Size, Position)).

%% === Object Manipulation ===
add_object(Id,Color,Shape,Material,Size) :-
    default_position(X,Y,Z),
    assertz(object(Id,Color,Shape,Material,Size,[X,Y,Z])).

delete_object(Id) :-
    retractall(object(Id,_,_,_,_,_)).

move_object(Id,DX,DY,DZ) :-
    object(Id,C,S,M,Si,[X,Y,Z]),
    NewX is X+DX, NewY is Y+DY, NewZ is Z+DZ,
    retract(object(Id,C,S,M,Si,[X,Y,Z])),
    assertz(object(Id,C,S,M,Si,[NewX,NewY,NewZ])).

list_objects :-
    forall(object(ID,C,S,M,Si,Pos),
           format('~w: ~w ~w ~w (~w) at ~w~n',[ID,C,S,M,Si,Pos])
    ).

default_position(0,0,0).

%% === Interpret Symbolic Commands ===
interpret(Command, Action) :-
    string_lower(Command, Lower),
    split_string(Lower, " ", "", Tokens),
    parse_command(Tokens, Action).

%% === Grammar Rules ===

% Move: natural description
parse_command(["move", "the", SizeStr, ColorStr, ShapeStr, "to", Dir],
              move(ID, DX, DY, DZ)) :-
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    atom_string(Direction, Dir),
    format_atom_id(Size, Color, Shape, ID),
    direction_delta(Direction, DX, DY, DZ).

% Move: by delta
parse_command(["move", "object", IdStr, "by", DXs, DYs, DZs],
              move(Id, DX, DY, DZ)) :-
    atom_string(Id, IdStr),
    number_string(DX, DXs),
    number_string(DY, DYs),
    number_string(DZ, DZs).

% Delete by object ID
parse_command(["delete", "object", IdStr],
              delete(Id)) :-
    atom_string(Id, IdStr).

% Delete by description
parse_command(["delete", "the", SizeStr, ColorStr, ShapeStr],
              delete(ID)) :-
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    format_atom_id(Size, Color, Shape, ID),
    delete_object(ID).

parse_command(["add", SizeStr, ColorStr, ShapeStr, MaterialStr],
              add(NewID, Color, Shape, Material, Size)) :-
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    atom_string(Material, MaterialStr),
    gensym(obj_, NewID),
    add_object(NewID, Color, Shape, Material, Size).


%% === Direction Mapping ===
direction_delta(left,  -1, 0, 0).
direction_delta(right,  1, 0, 0).
direction_delta(forward, 0, 0, -1).
direction_delta(backward, 0, 0, 1).
direction_delta(up,     0, 1, 0).
direction_delta(down,   0, -1, 0).

%% === ID Formatter ===
format_atom_id(Size, Color, Shape, ID) :-
    atomic_list_concat([Size, Color, Shape], '_', ID).

%% Fallback
parse_command(_, unknown_command).
