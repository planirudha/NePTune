% NEW (not in KDASP): relations taken from NePTune's relation matrices instead of position/3.
% Use in place of spatial.lp. NePTune's front/behind come from depth, so they cannot
% be recovered from image coordinates. Heads match spatial.lp, so relate.lp is unchanged.
left_of(ID,ID') :- spatial_evidence(left,ID,ID').
right_of(ID,ID') :- spatial_evidence(right,ID,ID').
in_front_of(ID,ID') :- spatial_evidence(front,ID,ID').
behind_of(ID,ID') :- spatial_evidence(behind,ID,ID').
