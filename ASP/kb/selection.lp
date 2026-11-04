% Selection: filter_*, unique (uniqueness constraint), and (intersection), or (union).
% Rules are verbatim from kdasp/perfect_theory.lp.
state(T+1,ID) :- unique(T), state(T,ID).
:- unique(T), state(T,ID), state(T,ID'), ID!=ID'.
state(T+1,ID) :- filter_large(T), state(T,ID), has_size(ID,large).
state(T+1,ID) :- filter_small(T), state(T,ID), has_size(ID,small).
state(T+1,ID) :- filter_gray(T), state(T,ID), has_color(ID,gray).
state(T+1,ID) :- filter_red(T), state(T,ID), has_color(ID,red).
state(T+1,ID) :- filter_blue(T), state(T,ID), has_color(ID,blue).
state(T+1,ID) :- filter_green(T), state(T,ID), has_color(ID,green).
state(T+1,ID) :- filter_brown(T), state(T,ID), has_color(ID,brown).
state(T+1,ID) :- filter_purple(T), state(T,ID), has_color(ID,purple).
state(T+1,ID) :- filter_cyan(T), state(T,ID), has_color(ID,cyan).
state(T+1,ID) :- filter_yellow(T), state(T,ID), has_color(ID,yellow).
state(T+1,ID) :- filter_metal(T), state(T,ID), has_material(ID,metal).
state(T+1,ID) :- filter_rubber(T), state(T,ID), has_material(ID,rubber).
state(T+1,ID) :- filter_sphere(T), state(T,ID), has_shape(ID,sphere).
state(T+1,ID) :- filter_cylinder(T), state(T,ID), has_shape(ID,cylinder).
state(T+1,ID) :- filter_cube(T), state(T,ID), has_shape(ID,cube).
state(T+1,ID) :- and(T,T'), state(T,ID), state(T',ID).
state(T+1,ID) :- or(T,T'), state(T,ID).
state(T+1,ID') :- or(T,T'), state(T',ID').
