% Integer comparison: equal_integer, less_than, greater_than.
% Rules are verbatim from kdasp/perfect_theory.lp.
bool(T+1,yes) :- equal_integer(T,T'), int(T,V), int(T',V'), V=V'.
bool(T+1,no) :- equal_integer(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- less_than(T,T'), int(T,V), int(T',V'), V<V'.
bool(T+1,no) :- less_than(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- greater_than(T,T'), int(T,V), int(T',V'), V>V'.
bool(T+1,no) :- greater_than(T,T'), not bool(T+1,yes).
