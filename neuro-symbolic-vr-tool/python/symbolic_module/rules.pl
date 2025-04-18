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

%% load_scene(+JSONFile)
%  Read CLEVR scene JSON and assert object/6 facts.
load_scene(File) :-
    retractall(object(_,_,_,_,_,_)),
    open(File, read, In),
    json_read_dict(In, Dict),
    close(In),
    Objects = Dict.get(objects),
    maplist(assert_object, Objects).

assert_object(Obj) :-
    Id        = Obj.get('id'),
    Color     = Obj.get('color'),
    Shape     = Obj.get('shape'),
    Material  = Obj.get('material'),
    Size      = Obj.get('size'),
    Position  = Obj.get('translation'),
    assertz(object(Id,Color,Shape,Material,Size,Position)).

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
    split_string(Command, " ", "", Tokens),
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

parse_command(["delete","object",IdStr],
              delete(IdAtom)) :-
    atom_number(IdAtom,IdStr),
    delete_object(IdAtom).

parse_command(["move","object",IdStr,"by",DXs,DYs,DZs],
              move(IdAtom,DX,DY,DZ)) :-
    atom_number(IdAtom,IdStr),
    number_string(DX,DXs),
    number_string(DY,DYs),
    number_string(DZ,DZs),
    move_object(IdAtom,DX,DY,DZ).

%% Fallback: no match
parse_command(_,unknown_command).
