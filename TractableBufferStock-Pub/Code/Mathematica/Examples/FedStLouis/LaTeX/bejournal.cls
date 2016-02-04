%%% LaTeX class for manuscripts submitted to the
%%% Berkeley Electronic Journal of Macroeconomics (BEPress)
%%%
%%% author: Christopher Carroll
%%% license: LaTeX Project Public License 
%%%
%%% Modified from style itaxpf by Arne Henningsen
%%% version: 1.0 (09 July 2007)
%%% license: LaTeX Project Public License 
%%%
%%% Note: This is an unofficial style file that may not fully comply
%%% with the Formatting Instructions of the BEPress Journals
%%% in every detail.
%%% Further information is available at
%%% http://www.uni-kiel.de/agrarpol/ahenningsen/latex-bibtex.html
%%%
%%% This document class is based on the "scrartcl" class
%%% from the KOMA script bundle.  For documentation, Google KOMA-Script Documentation
%%%
%%% The corresponding BibTeX style file "bejournal.bst" should be used
%%% for the bibliography: \bibliographystyle{bejournal}
%%%
%%% This class provides 2 options:
%%% titlepage = include a title page (with command "\maketitle")
%%% noendfloat = do not move "floats" (tables and figures) to the end
%%%
%%% This document class provides several new commands:
%%% If the titlepage option is invoked:
%%%     \keywords can be used to show keywords below the abstract
%%%     \jelclass can be used to show JEL classifications below the abstract

%%%     \thanks{text of the thank-you footnote} produces a footnote at the bottom of the page containing the text

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bejournal}[2012/03/11 LaTeX class for the Berkeley Electronic Journals, version 1.6]

\RequirePackage{ifthen}
\RequirePackage{chngpage}

% If unspecified whether pdf output (instead of dvi) is being used, determine that it SHOULD be
\newif\ifdvi\dvitrue
\@ifundefined{pdfoutput}{}{\ifnum\pdfoutput>0 \dvifalse\fi}

% permit double spacing
\RequirePackage{setspace}
%\doublespacing

% option to include a title page
\newcounter{IncludeTitlePage}
\setcounter{IncludeTitlePage}{0}
\DeclareOption{titlepage}{\setcounter{IncludeTitlePage}{1}}

% pass any options on to the scrartcl class and load this class with some options
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\ProcessOptions\relax
\LoadClass[fontsize=12pt,english,numbers=noenddot,captions=tableheading,captions=nooneline]{scrartcl}
% pointlessnumbers = do not add final dot after ordinary section numbers
% tablecaptionabove = use \captionabove for captions above rather than below tables
% noonelinecaption = do not treat one-line captions differently

% use new TeX encoding scheme (T1), which contains accented (European) letters,
% to make European hyphenations work
\RequirePackage[T1]{fontenc}

% use babel package and set language to english
\RequirePackage[english]{babel}

%% Amy change, Feb 28, 2011 ==>>
% spacing around formulas
\AtBeginDocument{
\setlength{\abovedisplayshortskip}{6pt}
\setlength{\belowdisplayshortskip}{6pt}
\setlength{\abovedisplayskip}{6pt}
\setlength{\belowdisplayskip}{6pt}
\allowdisplaybreaks % Permits LaTeX to break long formulas across pages (?)
}

%% Added the above info to the definition of normalsize
\def\normalsize{\@setfontsize \normalsize \@xiipt {14.5}
\abovedisplayskip 6\p@ \@plus 3\p@ \@minus 3\p@ 
\belowdisplayskip \abovedisplayskip 
\abovedisplayshortskip 6pt \belowdisplayshortskip 6pt
\let \@listi \@listI}
\normalsize
%% <<== End Amy change, Feb 28, 2011 

% Don't indent the second and later lines of footnotes
\deffootnote[1em]{0.0em}{1em}{\textsuperscript{\thefootnotemark}}

% spacing around captions and floats
\setlength{\abovecaptionskip}{6pt}
\setlength{\belowcaptionskip}{3pt}
\setlength{\floatsep}{0pt}
\setlength{\textfloatsep}{20pt}
\setlength{\intextsep}{15pt}

\RequirePackage{calc}

\newenvironment{Description}
{\begin{list}{}{\let\makelabel\Descriptionlabel
    \setlength\labelwidth{75pt}
    \setlength\labelsep{0pt}
    \setlength\leftmargin{75pt}
    \setlength\itemindent{0pt}
}}
{\end{list}}
\newcommand*\Descriptionlabel[1]{\textbf{#1}\hfil}

\RequirePackage{cancel,verbatim,
amsmath,amssymb,amsfonts,amsthm
,threeparttable,dcolumn,multicol,multirow,booktabs % table environment extensions
,latexsym
,afterpage
,endnotes
,moreverb
,hhline    % Allow double hlines
,tipa      % Define more characters, in particular the \textthorn character
,color
}

\RequirePackage{bejournal}
\RequirePackage[mathscr]{eucal}
\RequirePackage[normalem]{ulem}
\providecommand\bmmax{0} % Prevent boldmath package from using up too much memory
\RequirePackage{bm}      % Allow bold math
\RequirePackage{bbm}     % Blackbord math

% font for URLs
\RequirePackage{url}
\def\UrlFont{\ttfamily}

\usepackage[BE]{optional} % Enable the control of compilation using options passed on startup
\opt{EndFloats}{\RequirePackage[noheads,nolists,tablesfirst,nomarkers]{endfloat}}

% page format, margins if PrintGeom option is invoked
\opt{PrintGeom}{
\RequirePackage{geometry}
\geometry{letterpaper, tmargin=1.4in, bmargin=1.4in, lmargin=1.25in,
  rmargin=1.25in, headheight=0in, headsep=0in, footskip=0.5in }
}

\opt{JournalFormatting}{ % BEJM requires 1.5 inch margins
\RequirePackage{geometry}
\geometry{letterpaper, tmargin=1.5in, bmargin=1.5in, lmargin=1.5in,
  rmargin=1.5in, headheight=0in, headsep=0in, footskip=0.5in }
}

\providecommand{\textSizeDefault}{\normalsize}
\opt{bigWide}{\renewcommand{\textSizeDefault}{\large}}

\providecommand{\abstractSizeDefault}{\small}
\opt{bigWide}{\renewcommand{\abstractSizeDefault}{\normalsize}}

%% formatting section headers
\setkomafont{section}{\normalfont \Large \bfseries}
\setkomafont{subsection}{\normalfont \large}
\setkomafont{subsubsection}{\normalfont \itshape}

% citations and bibliography
\RequirePackage[authoryear]{natbib}
\AtBeginDocument{\renewcommand{\refname}{\clearpage References}
\renewcommand\@makefnmark
  {\mbox{\textsuperscript{\tiny\@thefnmark}}} % Restore footnote defn after titlepage redefines it
\ifdvi
  \renewcommand{\ushort}{\underline} % ushort seems to not work properly for tex4ht (which compiles using dvi)
\fi
}

% captions of figures and tables
\setcapwidth[c]{\textwidth}
% \setcapindent*{0pt}
\setkomafont{captionlabel}{\centering \bfseries}
\setkomafont{caption}{\centering}
\renewcommand{\captionformat}{~ }

% titlepage with title, author(s), and date
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
   \end{LARGE}\medskip
   {\normalsize \@date \par}       
   {\tiny{~}\par}
   \ifthenelse{\equal{\theIncludeTitlePage}{1}}{
      \vspace*{5mm}
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
   }{}
   \end{center}
}

\newcommand{\maketitleWithForcedDate}[1]{
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
   {\normalsize #1 \par}       
   {\tiny{~}\par}
   \ifthenelse{\equal{\theIncludeTitlePage}{1}}{
      \vspace*{5mm}
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
   \begin{spacing}{0.9}
   \noindent {\tiny \phantom{.}} \\ % Trick to get proper spacing in html
   \noindent \hrule height 0.4pt depth 0.0pt width \textwidth \relax
   \vspace*{4mm}
   \abstractSizeDefault
   \noindent \textbf{Abstract}\\
   \indent 
}{%
   \noindent {\tiny \phantom{.}} \\  % Trick to get proper spacing in html
   \noindent \hrule height 0.4pt depth 0.0pt width \textwidth \relax
   \vspace*{3mm}
   \ifthenelse{ \isundefined\@keywords }{
   \ClassWarningNoLine{bejournal}{No keywords specified.
      Please use the command \protect\keywords}
   }{
   }
   \end{spacing}

\begin{quote}
\begin{Description}
\item[\textbf{Keywords~}] \@keywords
\ifthenelse{ \isundefined\@jelclass }{
\ClassWarningNoLine{bejournal}{No JEL classification specified.
   Please use the command \protect\jelclass}
}{
\item[\textbf{JEL codes~}] \@jelclass
}
\ifthenelse{ \isundefined\@aspublished }{
}{
{

\vspace{0.1in}\centerline{\it \@aspublished}}
}
\end{Description}
\end{quote}
}

\setcounter{footnote}{0}%

\medskip

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
\RequirePackage{manyfoot,perpage}
\DefineFNsymbols{blank}[text]{{} {} {} {} {}}
\setfnsymbol{blank}
\DeclareNewFootnote{B}[fnsymbol]
\MakePerPage{footnoteB}


\renewcommand{\thanks}{\footnoteB}
\newcommand{\thanksFooter}{\footnoteB}

\hfuzz2pt % Don't bother to report over-full boxes if over-edge is < 2pt

%%%%%%%%%% Added by Amy Hendrickson, TeXnology Inc.
%% Amy change, Feb 28, 2011 ==>>

%%% Generalization of the author and authorinfo commands, based on
%%% the original code.

%% Instead of following author name with \authorAmark to \authorEmark,
%% please use \num, which can be used as many times as wanted.

%% Sample use:
%% \author{Christopher D. Carroll\num \\ 
%% \and
%% Jiri Slacalek\num \\ 
%% \and
%% Martin Sommer\num \\ 
%% \and
%% Another Name\num
%% \and
%% Still Another Name\num
%%  }

\newcounter{authornum}
\def\num{\global\advance\c@authornum by 1\textsuperscript{\tiny \the\c@authornum}}


%%%
%% Generalizaton of \authorsAinfo to \authorsABCDEinfo, new commands
%% \begin{authorsinfo}...\end{authorsinfo}

%% Now used:
%% \begin{authorsinfo}
%% \name{First author: info}
%% \\
%% \name{Second author: info}
%% \\
%% \name{Third author: info}
%% (and so on for as many authors
%% as you have)
%% \end{authorsinfo}
%%     (note: the \\ doesn't do anything, but makes it easier to
%%      see the separate author names.)

\newcounter{bottomauthor}
\def\authorsinfo{\global\c@bottomauthor=0\setbox0=\vbox\bgroup\footnotesize
\let\\ \relax
}

\def\endauthorsinfo{\egroup
\footnoteB{\unvbox0}
}

\def\name#1{\global\advance\c@bottomauthor by 1\textsuperscript{\tiny \the\c@bottomauthor}{#1}{~~~~}}

%% <<== End Amy change, Feb 28, 2011 

\endinput

