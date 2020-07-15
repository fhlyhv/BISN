% change the directories on lines 5, 7, and 8 to the corresponding directories 
% of Intel MLK in your OS
include_paths = {'-I./', ['-I', fullfile(pwd, './armadillo/include')], ...
    ['-I', fullfile(pwd, './armadillo/mex_interface')], ...
    '-I"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include"'};

library_paths = {'-L"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64_win"', ...
    '-L"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64_win"'};

COMPFLAGS = 'COMPFLAGS="$COMPFLAGS /DMKL_ILP64 /openmp /O2 /GL"';
LDFLAGS = 'LDFLAGS="$LDFLAGS /LTCG:INCREMENTAL"';

libraries = {'-lmkl_intel_lp64', '-lmkl_intel_thread', '-lmkl_core', '-llibiomp5md'};

mex(include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, 'BISN.cpp');

mex(include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, 'BISN_missing.cpp');

mex(include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, 'QUICParameterLearning.cpp');