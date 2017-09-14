% Set Program Path
function	set_dir

%root_dir = getenv('MFILE');
%prog_dir = [root_dir '/VBtool/prediction'];
prog_dir = pwd;

addpath([prog_dir]);
addpath([prog_dir '/test']);
addpath([prog_dir '/basic_func']);
addpath([prog_dir '/../mex_prog']);
