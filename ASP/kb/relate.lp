% relate_* steps: objects standing in a relation to the current state.
% Rules are verbatim from kdasp/perfect_theory.lp.
state(T+1,ID) :- relate_left(T), state(T,ID'), left_of(ID,ID').
state(T+1,ID) :- relate_right(T), state(T,ID'), right_of(ID,ID').
state(T+1,ID) :- relate_front(T), state(T,ID'), in_front_of(ID,ID') .
state(T+1,ID) :- relate_behind(T), state(T,ID'), behind_of(ID,ID').
