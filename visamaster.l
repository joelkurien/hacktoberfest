/*the lex program for finding the appropriate account numbers for visa and master card */
%{
#include<stdio.h>
#include<string.h>
%}


%%
^[4][0-9]+ {
    if(yyleng==12)
        printf("Visa\n");
}

^[5][0-5][0-9]+ {
    if(yyleng==16)
        printf("MasterCard\n");
}

[0-9]+ {
    printf("Invalid\n");
}
%%

int yywrap(){}

int main()
{
    yylex();
    return 0;
}
