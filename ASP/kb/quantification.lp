% Quantification: count, exist, boolean_negation.
% Rules are verbatim from kdasp/perfect_theory.lp.
int(T+1,V) :- count(T), #count{ ID : state(T,ID) } = V.
bool(T+1,yes) :- exist(T), state(T,ID).
bool(T+1,no) :- exist(T), not bool(T+1,yes).
bool(T+1, yes) :- boolean_negation(T), bool(T, no).
bool(T+1, no) :- boolean_negation(T), not bool(T+1, yes).
