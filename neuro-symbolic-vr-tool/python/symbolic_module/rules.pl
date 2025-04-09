:- dynamic object/3.  % object(ID, Type, Position)
default_pos([0, 0, 0]).

% Helper: generate ID based on type and count
generate_id(Type, ID) :-
    aggregate_all(count, object(_, Type, _), Count),
    atom_concat(Type, Count, ID).

% Create shape if no duplicate ID exists
action(create(Type)) :-
    generate_id(Type, ID),
    \+ object(ID, Type, _),
    default_pos(Pos),
    assertz(object(ID, Type, Pos)),
    format('Created ~w (~w) at ~w~n', [Type, ID, Pos]), !.

% Prevent duplicate of same ID
action(create(Type)) :-
    format('Could not create another ~w~n', [Type]), !.

% Delete by type (removes the latest one of that shape)
action(delete(Type)) :-
    object(ID, Type, Pos),
    retract(object(ID, Type, Pos)),
    format('Deleted ~w (~w)~n', [Type, ID]), !.

action(delete(Type)) :-
    format('No ~w to delete.~n', [Type]), !.

% Move the most recent object of that type
action(move(Type, Pos)) :-
    object(ID, Type, _),
    retract(object(ID, Type, _)),
    assertz(object(ID, Type, Pos)),
    format('Moved ~w (~w) to ~w~n', [Type, ID, Pos]), !.

action(move(Type, _)) :-
    format('Cannot move ~w: does not exist.~n', [Type]), !.

% View current state
list_objects :-
    forall(object(ID, Type, Pos),
           format('Object ~w: ~w at ~w~n', [ID, Type, Pos])).

% Fallback
action(X) :-
    format('Unrecognized action: ~w~n', [X]).
