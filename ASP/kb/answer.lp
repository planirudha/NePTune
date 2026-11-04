% Answer derivation: ans/1 from the value at the end(T) step.
% Rules are verbatim from kdasp/perfect_theory.lp.
ans(V) :- end(T), size(T,V).
ans(V) :- end(T), color(T,V).
ans(V) :- end(T), material(T,V).
ans(V) :- end(T), shape(T,V).
ans(V) :- end(T), bool(T,V).
ans(V) :- end(T), int(T,V).
:- not ans(_).
#show ans/1.
