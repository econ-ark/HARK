%%
%% This is file `expcites.bst',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% bibexport.dtx  (with options: `expcites')
%%
%% (c) 2019/03/30 Nicolas Markey <bibexport at markey dot fr>
%%
%% This work may  be distributed and/or modified under  the conditions of
%% the LaTeX Project  Public License, either version 1.3  of this license
%% or (at  your option)  any later version.   The latest version  of this
%% license is in
%%
%%   http://www.latex-project.org/lppl.txt
%%
%% and version 1.3 or later is part of all distributions of LaTeX version
%% 2005/12/01 or later.
%%
%% This work has the LPPL maintenance status `maintained'.
%% The Current Maintainer of this work is Nicolas Markey.
%%
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%
FUNCTION{left.width}{#23}
FUNCTION{right.width}{#55}
FUNCTION{url.right.width}{#61}
FUNCTION{left.short.width}{#10} %% for @preamble
FUNCTION{right.long.width}{#63}
FUNCTION{left.delim}{quote$}
FUNCTION{right.delim}{quote$}
ENTRY{dummy}{}{}
FUNCTION{not}
{
    {#0}
    {#1}
  if$
}
FUNCTION{and}
{
    'skip$
    {pop$ #0}
  if$
}
FUNCTION{or}
{
    {pop$ #1}
    'skip$
  if$
}
INTEGERS{left.length right.length}
STRINGS{ s t }
INTEGERS{bool cpt}
FUNCTION{space.complete}
{
  'left.length :=
  duplicate$ text.length$ left.length swap$ -
  {duplicate$ #0 >}
    {
      swap$ " " * swap$ #1 -
    }
  while$
  pop$
}
FUNCTION{split.string}
{
  'right.length :=
  duplicate$ right.length #1 + #1 substring$ "" =
    {""}
    {
      's :=
      right.length
      {duplicate$ duplicate$ s swap$ #1 substring$ " " = not and}
        {#1 -}
      while$
      duplicate$ #2 <
        {
          pop$ "    " s * ""
        }
        {
          duplicate$ s swap$ #1 swap$ substring$
          swap$
          s swap$ global.max$ substring$
        }
      if$
    }
  if$
}
FUNCTION{init.cited.keys}
{
  left.delim 's :=
  #0 'bool :=
  #0 'cpt :=
}
FUNCTION{write.cited.keys}
{
  bool
    {"" left.width space.complete swap$}
    {"  list of keys: " left.width space.complete swap$
     #1 'bool :=}
  if$
  {duplicate$ text.length$ right.width >}
    {
      right.width split.string 't :=
      *
      write$ newline$
      "" left.width space.complete t
    }
  while$
  pop$ pop$ t
}
FUNCTION{write.cited.keys.last}
{
  bool
    {"" left.width space.complete swap$}
    {"  list of keys: " left.width space.complete swap$
     #1 'bool :=}
  if$
  {duplicate$ duplicate$ text.length$ #1 substring$ "," = not}
    {duplicate$ text.length$ #1 - #1 swap$ substring$}
  while$
  duplicate$ text.length$ #1 - #1 swap$ substring$
  right.delim * "," *
  {duplicate$ "" = not}
    {
      right.width split.string 't :=
      *
      write$ newline$
      "" left.width space.complete t
    }
  while$
  pop$ pop$
}
FUNCTION{write.nbkeys}
{
  "  number of entries: " left.width space.complete
  " " *
  cpt int.to.str$ * "," * write$ newline$
}
FUNCTION{cited.keys}
{
  cpt #1 + 'cpt :=
  s cite$ ", " * *  's :=
  s text.length$ #4000 >
    {s write.cited.keys 's :=}
    'skip$
  if$
}
FUNCTION{end.cited.keys}
{
  s write.cited.keys.last
  write.nbkeys
}
FUNCTION{article}{cited.keys}
FUNCTION{book}{cited.keys}
FUNCTION{booklet}{cited.keys}
FUNCTION{conference}{cited.keys}
FUNCTION{habthesis}{cited.keys}
FUNCTION{inbook}{cited.keys}
FUNCTION{incollection}{cited.keys}
FUNCTION{inproceedings}{cited.keys}
FUNCTION{journals}{cited.keys}
FUNCTION{manual}{cited.keys}
FUNCTION{mastersthesis}{cited.keys}
FUNCTION{misc}{cited.keys}
FUNCTION{phdthesis}{cited.keys}
FUNCTION{proceedings}{cited.keys}
FUNCTION{techreport}{cited.keys}
FUNCTION{unpublished}{cited.keys}
READ
EXECUTE{init.cited.keys}
ITERATE{cited.keys}
EXECUTE{end.cited.keys}
%%
%%
%% End of file `expcites.bst'.
