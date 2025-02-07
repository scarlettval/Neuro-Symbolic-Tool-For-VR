:- initialization(main).

main :- 
    absolute_file_name('symbolic_rules.pl', Path),
    format('Resolved Path: ~w~n', [Path]).

