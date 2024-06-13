%%% LaTeX class for econark
%%%
%%% author: Christopher Carroll
%%% license: LaTeX Project Public License
%%%
%%% Modified from style itaxpf by Arne Henningsen
%%% version: 1.0 (09 July 2007)
%%% license: LaTeX Project Public License
%%%
%%% Further information is available at
%%% http://www.uni-kiel.de/agrarpol/ahenningsen/latex-bibtex.html
%%%
%%% This document class is based on the "scrartcl" class
%%% from the KOMA script bundle.  For documentation, Google KOMA-Script Documentation
%%%
%%% The corresponding BibTeX style file "econark.bst" should be used
%%% for the bibliography: \bibliographystyle{econark}
%%%
%%% In addition to the options for scrartcl, this package includes an additional option
%%% titlepage = include a title page (with command "\maketitle")
%%%
%%% If an abstract is defined by \begin{abstract}\end{abstract} before \maketitle is invoked,
%%% it will appear on the title page
%%%
%%% If the titlepage option is invoked:
%%%     \keywords can be used to show keywords below the abstract
%%%     \jelclass can be used to show JEL classifications below the abstract
%%%     \thanks{text of the thank-you footnote} produces a footnote at the bottom of the page containing the text

\ProvidesClass{econark}[2024/06/04 LaTeX class for econark latex documents by Christopher Carroll]
\RequirePackage{snapshot} % Creates list of external dependencies used by a LaTeX document
\RequirePackage{ifthen}   % allows conditional execution
%\RequirePackage{rotating}
\RequirePackage{changepage}
\RequirePackage{currfile} % define macros to produce the filename and path

% If unspecified whether pdf output (instead of dvi) is being used, determine that it SHOULD be
\newif\ifdvi\dvitrue
\@ifundefined{pdfoutput}{}{\ifnum\pdfoutput>0 \dvifalse\fi}

% option to include a titlepage needs to be set up before class is loaded
\newcounter{IncludeTitlePage}
\setcounter{IncludeTitlePage}{0}
\DeclareOption{titlepage}{\setcounter{IncludeTitlePage}{1}}

% pass on any options with which econark was invoked
% to the scrartcl class and load this class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\ProcessOptions\relax

% see documentation for KOMA class for meanings of these options
\LoadClass[
fontsize=12pt,english,
numbers=noenddot,          % do not add final dot after ordinary section numbers
captions=tableheading,     % allows \captionabove for captions above rather than below tables
captions=nooneline         % do not treat one-line captions differently
,headings=optiontocandhead % section headings at top of page
]{scrartcl}

% use TeX encoding scheme (T1), which contains accented (European) letters,
% to make European hyphenations work
\RequirePackage[T1]{fontenc}

% use babel package to set language to english
\RequirePackage[english]{babel}

% % spacing around formulas
% \AtBeginDocument{
%   \setlength{\abovedisplayshortskip}{6pt}
%   \setlength{\belowdisplayshortskip}{6pt}
%   \setlength{\abovedisplayskip}{6pt}
%   \setlength{\belowdisplayskip}{6pt}
%   \allowdisplaybreaks % Permits LaTeX to break long formulas across pages (?)
% }

% Don't indent the second and later lines of footnotes
\deffootnote[1em]{0.0em}{1em}{\textsuperscript{\thefootnotemark}}

% % spacing around captions and floats
% \setlength{\abovecaptionskip}{6pt}
% \setlength{\belowcaptionskip}{3pt}
% \setlength{\floatsep}{0pt}
% \setlength{\textfloatsep}{20pt}
% \setlength{\intextsep}{15pt}

\RequirePackage{amsmath,amssymb,amsfonts,amsthm % math stuff
  ,cancel    % \cancel{thing} does strikethrough
  ,verbatim  % do not parse text in verbatim environment
  ,dcolumn   % align on decimal
  ,hhline    % Allow double hlines
  ,xcolor    % Superceded color, which is WAY out of date
  ,accents   % More math accents
  ,appendix  % handles appendices
  ,datetime2 % create and manipulate dates
}
% to get endnotes with superscripts, use \printendnotes[superscriptednotes]

% ---------------Custom Definitions--------------

\def\liminf{\mathop{\rm lim\, inf}\limits}
%\DeclareMathOperator{\Ex}{\mathbb{E}} % Expectations operator
\def\var{\operatorname{var}} % variance
\def\cov{\operatorname{cov}} % covariance
\def\std{\operatorname{std}} % standard deviation
\def\argmax{\operatornamewithlimits{arg\,max}} % argmax

% ---------------Custom Modifications----------------------

% \htline defines a thick horizontal line in table
\def\htline{\noalign{\hrule height 2\arrayrulewidth}}

% \hdline defines a dotted horizontal line in table
\def\hdline{\noalign{.\dotfill{}.}}

% To keep footnotes on a single page except in extreme cases
\interfootnotelinepenalty=5000 %from 0 to 10000

% To prevent hyphenation
\hyphenpenalty=5000 %from 0 to 10000, default is 200

% To prevent breaking math formula
\relpenalty=4000 %from 0 to 10000, default is 500

% To prevent breaking math formula after binary operator
\binoppenalty=10000 %from 0 to 10000, default is 700

% --------------------------------------------------------

\RequirePackage[mathscr]{eucal} % enable mathscr and mathcal fonts
%\RequirePackage[normalem]{ulem}
%\providecommand\bmmax{0} % Prevent boldmath package from using up too much memory
%\RequirePackage{bm}      % Allow bold math
%\RequirePackage{bbm}     % Blackbord math

% font for URLs
\RequirePackage{url}
\def\UrlFont{\ttfamily} % typewriter

\usepackage[econtex]{optional} % Enable the control of compilation using options passed on startup
% To put floats (tables, figs) at end, include option endfloat in pdflatex
\opt{EndFloats}{\RequirePackage[noheads,nolists,tablesfirst,nomarkers]{endfloat}}

\providecommand{\textSizeDefault}{\normalsize}
\providecommand{\abstractSizeDefault}{\small}

%% formatting section headers
\setkomafont{section}{\normalfont \Large \bfseries}
\setkomafont{subsection}{\normalfont \large}
\setkomafont{subsubsection}{\normalfont \itshape}

% citations and bibliography
\RequirePackage[authoryear]{natbib}

% Configure references at startup
\AtBeginDocument{\renewcommand{\refname}{References}
  \renewcommand\@makefnmark
  {\mbox{\textsuperscript{\tiny\@thefnmark}}} % Restore footnote defn after titlepage redefines it
  \ifdvi
  \else % \ifdvi failed so we must be in pdf mode
  % \providecommand{\ushort}{}
  % \renewcommand{\ushort}{\underline} % ushort seems to not work properly for tex4ht (which compiles using dvi)
  \fi
}

% captions of figures and tables
\setcapwidth[c]{\textwidth}

\setkomafont{captionlabel}{\centering \bfseries}
\setkomafont{caption}{\centering}
\renewcommand{\captionformat}{~ }

% Title page stuff (allows keywords and JEL codes)
\newenvironment{Description}
{\begin{list}{}{\let\makelabel\Descriptionlabel
      \setlength\labelwidth{75pt}
      \setlength\labelsep{0pt}
      \setlength\leftmargin{75pt}
      \setlength\itemindent{0pt}
    }}
  {\end{list}}
\newcommand*\Descriptionlabel[1]{\textbf{#1}\hfil}

\renewcommand{\maketitle}{
  \ifthenelse{\equal{\theIncludeTitlePage}{1}}{
    \let\footnotesize\scriptsize
    \thispagestyle{empty}
    \vspace*{0mm}
    \enlargethispage{0.4in}
  }{}
  \begin{center}
    \begin{LARGE}
      \begin{spacing}{1.00}
        \textbf{\@title}
      \end{spacing}
    \end{LARGE}\medskip\medskip\medskip
    {\normalsize \@date \par}
    {\tiny{~}\par}
    \ifthenelse{\equal{\theIncludeTitlePage}{1}}{
      \vspace*{3mm}
      {\large
        \lineskip .75em%
        \begin{tabular}[t]{c}%
          \@author
        \end{tabular}\par}%
    }{}
  \end{center}
}

% abstract, keywords, JEL classification
\newcommand{\keywords}[1]{\newcommand{\@keywords}{#1}}
\newcommand{\jelclass}[1]{\newcommand{\@jelclass}{#1}}
\newcommand{\aspublished}[1]{\newcommand{\@aspublished}{#1}}

\renewenvironment{abstract}{%
    \abstractSizeDefault
    \noindent \textbf{Abstract}\\
    \indent
  }{%
%    \noindent {\tiny \phantom{.}} \\  % Trick to get proper spacing in html
    \ifdvi
    \else
%    \noindent \hrule height 0.4pt depth 0.0pt width \textwidth \relax
    \fi
    \vspace*{0mm}
    \ifthenelse{ \isundefined\@keywords }{
      \ClassWarningNoLine{econtex}{No keywords specified.
        Please use the command \protect\keywords}
    }{
    }
%  \end{spacing}

  \begin{quote}
    \begin{Description}
    \item[\textbf{Keywords~}] \@keywords
      \ifthenelse{ \isundefined\@jelclass }{
        \ClassWarningNoLine{econtex}{No JEL classification specified.
          Please use the command \protect\jelclass}
      }{
      \item[\textbf{JEL codes~}] \@jelclass
      }

    \end{Description}
    \ifthenelse{ \isundefined\@aspublished }{
    }{
      {

        \vspace{0.1in}\@aspublished}
    }\end{quote}
}

\setcounter{footnote}{0}%

% avoid clubs and widows
\clubpenalty=1000  % penalty for breaking a page with only one sentence; set to 10000 to prohibit
\widowpenalty=10000
% \displaywidowpenalty=10000

\hbadness 3000  % LaTeX default is 1000, which is too finicky - set to 10000 to get rid of complaints

% increase stretchability of the spaces (avoid overfull hboxes)
\setlength{\emergencystretch}{3em}

\pagestyle{plain}
\thispagestyle{empty}

\newcommand{\titlepagefinish}{\newpage
  \renewcommand\@makefnmark
  {\mbox{\textsuperscript{\tiny\@thefnmark}}}
}

\RequirePackage[multiple]{footmisc}
\RequirePackage{manyfoot}
\RequirePackage{perpage} \DefineFNsymbols{blank}[text]{{} {} {} {} {}} \setfnsymbol{blank} \DeclareNewFootnote{B}[fnsymbol] \MakePerPage{footnoteB} \renewcommand{\thanks}{\footnoteB} \newcommand{\thanksFooter}{\footnoteB}

\hfuzz2pt % Don't bother to report over-full boxes if over-edge is < 2pt

\RequirePackage{setspace} % needed for the titlepage stuff below

% Titlepage
%%%%%%%%%% Added by Amy Hendrickson, TeXnology Inc.
%% Amy change, Feb 28, 2011 ==>>

%%% Generalization of the author and authorinfo commands, based on
%%% the original code.

%% Sample use:
%% \author{Christopher D. Carroll\authNum \\
%% \and
%% Jiri Slacalek\authNum \\
%% \and
%% Martin Sommer\authNum \\
%% \and
%% Another Name\authNum
%% \and
%% Still Another Name\authNum
%% }

\newcounter{authornum} % Use either \num or (preferred) \authNum to denote authors
\def\num{\global\advance\c@authornum by 1\textsuperscript{\tiny \the\c@authornum}}
\def\authNum{\global\advance\c@authornum by 1\textsuperscript{\tiny \the\c@authornum}}

%% \begin{authorsinfo}...\end{authorsinfo}

%% Now use:
%% \begin{authorsinfo}
%%   \name{First author: info}
%%   \\
%%   \name{Second author: info}
%%   \\
%%   \name{Third author: info}
%%   (and so on for as many authors
%%   as you have)
%% \end{authorsinfo}
%% (note: the \\ doesn't do anything, but makes it easier to
%% see the separate author names.)

\newcounter{bottomauthor}
\def\authorsinfo{\global\c@bottomauthor=0\setbox0=\vbox\bgroup\footnotesize
  \let\\ \relax
}

\def\endauthorsinfo{\egroup
  \footnoteB{\unvbox0}
}

\def\name#1{\global\advance\c@bottomauthor by 1\textsuperscript{\tiny \the\c@bottomauthor}{#1}{~~~~}}

\makeatletter
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\makeatother
