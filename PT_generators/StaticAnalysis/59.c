int main() {
  // variable declarations
  int c;
  int n;
  // pre-conditions
  (c = 0);
  assume((n > 0));
  // loop body
  if(unknown()){
  }
  while (unknown()) {
    {
        int j = 0;
      if ( c > 18 ) {
        if ( (c != n) )
        {
        (c  = (c + 1));
        }
      } else {
        if ( (c == 999) )
        {
        (c  += j);
        }
      }

    }
  }

  // post-condition
if ( (c != n) )
assert( (c <= n) );
}
