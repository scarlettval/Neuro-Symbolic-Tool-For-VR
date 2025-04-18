%% rules.pl
%% Dynamic scene management and command interpretation for CLEVR-based VR.

:- module(rules, [
    load_scene/1,
    interpret/2,
    add_object/5,
    delete_object/1,
    move_object/4,
    list_objects/0
]).

:- use_module(library(http/json)).   % json_read_dict/2
:- dynamic object/6.
:- discontiguous parse_command/2.

%% load_scene(+JSONFile)
%  Read CLEVR scene JSON and assert object/6 facts.
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

%% add_object(+ID, +Color, +Shape, +Material, +Size)
%  Default at origin.
add_object(Id,Color,Shape,Material,Size) :-
    default_position(X,Y,Z),
    assertz(object(Id,Color,Shape,Material,Size,[X,Y,Z])).

%% delete_object(+ID)
delete_object(Id) :-
    retractall(object(Id,_,_,_,_,_)).

%% move_object(+ID, +DX, +DY, +DZ)
move_object(Id,DX,DY,DZ) :-
    object(Id,C,S,M,Si,[X,Y,Z]),
    NewX is X+DX, NewY is Y+DY, NewZ is Z+DZ,
    retract(object(Id,C,S,M,Si,[X,Y,Z])),
    assertz(object(Id,C,S,M,Si,[NewX,NewY,NewZ])).

%% list_objects
list_objects :-
    forall(object(ID,C,S,M,Si,Pos),
           format('~w: ~w ~w ~w (~w) at ~w~n',[ID,C,S,M,Si,Pos])
    ).

%% default_position(-X,-Y,-Z)
default_position(0,0,0).

%% interpret(+CommandString, -ActionTerm)
%  Parses a space-separated command into an action.
interpret(Command, Action) :-
    string_lower(Command, Lower),
    split_string(Lower, " ", "", Tokens),
    parse_command(Tokens, Action).

%% Simple grammar for add, delete, move
parse_command(["add",Color,Shape,Size,Material],
              add(NewID,ColorAtom,ShapeAtom,MaterialAtom,SizeAtom)) :-
    atom_string(ColorAtom,Color),
    atom_string(ShapeAtom,Shape),
    atom_string(SizeAtom,Size),
    atom_string(MaterialAtom,Material),
    gensym(obj_,NewID),
    add_object(NewID,ColorAtom,ShapeAtom,MaterialAtom,SizeAtom).

parse_command(["move", "the", SizeStr, ColorStr, ShapeStr, "to", Dir1],
              move(ID, DX, DY, DZ)) :-
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    atom_string(Direction, Dir1),
    format_atom_id(Size, Color, Shape, ID),
    direction_delta(Direction, DX, DY, DZ),
    move_object(ID, DX, DY, DZ).

parse_command(["move", "the", SizeStr, ColorStr, ShapeStr, "to", "the", Dir2],
              move(ID, DX, DY, DZ)) :-
    atom_string(Size, SizeStr),
    atom_string(Color, ColorStr),
    atom_string(Shape, ShapeStr),
    atom_string(Direction, Dir2),
    format_atom_id(Size, Color, Shape, ID),
    direction_delta(Direction, DX, DY, DZ),
    move_object(ID, DX, DY, DZ).

parse_command(["delete","object",IdStr],
              delete(IdAtom)) :-
    atom_string(IdAtom,IdStr),
    delete_object(IdAtom).

parse_command(["move","object",IdStr,"by",DXs,DYs,DZs],
              move(IdAtom,DX,DY,DZ)) :-
    atom_string(IdAtom,IdStr),
    number_string(DX,DXs),
    number_string(DY,DYs),
    number_string(DZ,DZs),
    move_object(IdAtom,DX,DY,DZ).

%% direction_delta(+Direction, -DX, -DY, -DZ)
direction_delta(left,  -1, 0, 0).
direction_delta(right,  1, 0, 0).
direction_delta(forward, 0, 0, -1).
direction_delta(backward, 0, 0, 1).
direction_delta(up,     0, 1, 0).
direction_delta(down,   0, -1, 0).

%% format_atom_id(+Size, +Color, +Shape, -ID)
format_atom_id(Size, Color, Shape, ID) :-
    atomic_list_concat([Size, Color, Shape], '_', ID).

%% Fallback: no match
parse_command(_,unknown_command).