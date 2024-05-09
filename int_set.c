
#include "int_set.h"

//

typedef struct int_set {
    int             length, capacity;
    int_range_t     *elements;
} int_set_t;

//

int_set_t*
__int_set_create()
{
    int_set_t       *S = (int_set_t*)malloc(sizeof(int_set_t));
    
    if ( S ) {
        S->length = S->capacity = 0;
        S->elements = NULL;
    }
    return S;
}

//

void
__int_set_destroy(
    int_set_t   *S
)
{
    if ( S->elements ) free((void*)S->elements);
    free((void*)S);
}

//

bool
__int_set_grow(
    int_set_t   *S
)
{
    int             new_capacity = S->capacity + 64;
    int_range_t     *new_elements = (int_range_t*)realloc(S->elements, new_capacity * sizeof(int_range_t));
    
    if ( new_elements ) {
        S->elements = new_elements;
        S->capacity = new_capacity;
        return true;
    }
    return false;
}

//
////
//

int_set_ref
int_set_create()
{
    return (int_set_ref)__int_set_create();
}

//

void
int_set_destroy(
    int_set_ref S
)
{
    __int_set_destroy((int_set_t*)S);
}

//

int
int_set_get_length(
    int_set_ref S
)
{
    int         l = 0, ri = 0;
    
    while ( ri < S->length ) l += S->elements[ri++].length;
    return l;
}

//

bool
int_set_push_int(
    int_set_ref S,
    int         i
)
{
    int         ri = 0;
    
    // Check existing ranges for overlap/extension:
    while ( ri < S->length ) {
        // Already exists?
        if ( int_range_does_contain(S->elements[ri], i) ) return true;
        
        // Does the number occur immediately before this range?
        if ( i + 1 == S->elements[ri].start ) {
            // Extend this range:
            S->elements[ri].start--, S->elements[ri].length++;
            
            // Check if it now overlaps the range occurring before it:
            if ( (ri > 0) && int_range_is_adjacent_or_intersecting(S->elements[ri-1], S->elements[ri]) ) {
                // Replace with single range:
                S->elements[ri-1] = int_range_union(S->elements[ri-1], S->elements[ri]);
                
                // Shift anything else down:
                if ( ri + 1 < S->length )
                    memmove(&S->elements[ri], &S->elements[ri+1], sizeof(int_range_t) * (S->length - ri - 1));
                S->length--;
            }
            return true;
        }
        
        // Does the number occur immediately after this range?
        if ( S->elements[ri].start + S->elements[ri].length == i ) {
            // Extend this range:
            S->elements[ri].length++;
            
            // Check if it now overlaps the range occurring after it:
            if ( (ri + 1 < S->length) && int_range_is_adjacent_or_intersecting(S->elements[ri], S->elements[ri+1]) ) {
                // Replace with single range:
                S->elements[ri] = int_range_union(S->elements[ri], S->elements[ri+1]);
                
                // Shift anything else down:
                if ( ri + 2 < S->length )
                    memmove(&S->elements[ri+1], &S->elements[ri+2], sizeof(int_range_t) * (S->length - ri - 2));
                S->length--;
            }
            return true;
        }
        
        // Is the number ordered before this range?  If so, we found our
        // insertion point:
        if ( i < S->elements[ri].start ) break;
        
        ri++;
    }
    
    // New range necessary at index ri.  Start by making room for another
    // range if necessary:
    if ( S->length == S->capacity ) {
        if ( ! __int_set_grow(S) ) return false;
    }
    
    // Do we need to shift any latter ranges up to make room?
    if ( ri < S->length ) {
        memmove(&S->elements[ri+1], &S->elements[ri], sizeof(int_range_t) * (S->length - ri));
    }
    S->length++;
    
    // Insert the new range:
    S->elements[ri] = int_range_make_with_low_and_high(i, i);
    return true;
}

//

bool
int_set_push_range(
    int_set_ref S,
    int_range_t r
)
{
    int         ri = 0;
    
    // Check existing ranges for overlap/extension:
    while ( ri < S->length ) {
        // Do we have overlap?
        if ( int_range_does_intersect(S->elements[ri], r) ) {
            int     end_r  = r.start + r.length - 1;
            int     end_ri = S->elements[ri].start + S->elements[ri].length - 1;
            bool    ordered_asc;
            
            // Is r contained fully within the range?
            if ( r.start >= S->elements[ri].start && end_r <= end_ri ) return true;
            
            // Is r ordered before this range?
            ordered_asc = ( r.start < S->elements[ri].start );
            
            // Merge the new range into this one:
            S->elements[ri] = int_range_union(r, S->elements[ri]);
            
            // If r was before this range, we don't need to check for
            // merging-down with any previous range; but if r was
            // ordered after this range, we need to see if the next
            // range in S now overlaps:
            if ( ! ordered_asc && (ri + 1 < S->length) ) {
                if ( int_range_is_adjacent_or_intersecting(S->elements[ri], S->elements[ri+1]) ) {
                    // Merge the two ranges:
                    S->elements[ri] = int_range_union(S->elements[ri], S->elements[ri+1]);
                    
                    // Shift anything else down:
                    if ( ri + 2 < S->length )
                        memmove(&S->elements[ri+1], &S->elements[ri+2], sizeof(int_range_t) * (S->length - ri - 2));
                    S->length--;
                }
            }
            return true;
        }
        
        // Does r sit just before this range?
        if ( r.start + r.length == S->elements[ri].start ) {
            S->elements[ri] = int_range_union(r, S->elements[ri]);
            
            // Does the new range now overlap with or directly precede the
            // following range?
            if ( ri + 1 < S->length && int_range_is_adjacent_or_intersecting(S->elements[ri], S->elements[ri+1]) ) {
                // Merge the two ranges:
                S->elements[ri] = int_range_union(S->elements[ri], S->elements[ri+1]);
                
                // Shift anything else down:
                if ( ri + 2 < S->length )
                    memmove(&S->elements[ri+1], &S->elements[ri+2], sizeof(int_range_t) * (S->length - ri - 2));
                S->length--;
            }
            return true;
        }
        
        // Does r sit just after this range?
        if ( S->elements[ri].start + S->elements[ri].length == r.start ) {
            S->elements[ri] = int_range_union(S->elements[ri], r);
            
            // Does the new range now overlap with or directly precede the
            // following range?
            if ( ri + 1 < S->length && int_range_is_adjacent_or_intersecting(S->elements[ri], S->elements[ri+1]) ) {
                // Merge the two ranges:
                S->elements[ri] = int_range_union(S->elements[ri], S->elements[ri+1]);
                
                // Shift anything else down:
                if ( ri + 2 < S->length )
                    memmove(&S->elements[ri+1], &S->elements[ri+2], sizeof(int_range_t) * (S->length - ri - 2));
                S->length--;
            }
            return true;
        }
        
        // Is the range ordered before this range?  If so, we found our
        // insertion point:
        if ( r.start < S->elements[ri].start ) break;
        
        ri++;
    }
    
    // New range necessary at index ri.  Start by making room for another
    // range if necessary:
    if ( S->length == S->capacity ) {
        if ( ! __int_set_grow(S) ) return false;
    }
    
    // Do we need to shift any latter ranges up to make room?
    if ( ri < S->length ) {
        memmove(&S->elements[ri+1], &S->elements[ri], sizeof(int_range_t) * (S->length - ri));
    }
    S->length++;
    
    // Insert the new range:
    S->elements[ri] = r;
    return true;
}

//

bool
int_set_remove_int(
    int_set_ref S,
    int         i
)
{
    int         ri = 0;
    
    // Check ranges for inclusion of i:
    while ( ri < S->length ) {
        bool    did_contract = false;
        
        // In this range?
        if ( int_range_does_contain(S->elements[ri], i) ) {
            // Does the range start with this number?
            if ( S->elements[ri].start == i ) {
                S->elements[ri].start++;
                S->elements[ri].length--;
                did_contract = true;
            }
            // Does the range end with this number?
            else if ( i == S->elements[ri].start + S->elements[ri].length - 1 ) {
                S->elements[ri].length--;
                did_contract = true;
            }
            if ( did_contract ) {
                // Make sure the range is NOT zero length now:
                if ( S->elements[ri].length == 0 ) {
                    // Move everything else over this element:
                    if ( ri + 1 < S->length ) {
                        memmove(&S->elements[ri], &S->elements[ri+1], sizeof(int_range_t) * (S->length - ri - 1));
                    }
                    S->length--;
                }
            } else {
                // The range needs to be broken into two ranges:
                if ( S->length == S->capacity ) {
                    if ( ! __int_set_grow(S) ) return false;
                }
                if ( ri + 1 < S->length ) {
                    memmove(&S->elements[ri+2], &S->elements[ri+1], sizeof(int_range_t) * (S->length - ri - 1));
                }
                S->elements[ri+1] = int_range_make_with_low_and_high(i + 1, S->elements[ri].start + S->elements[ri].length - 1);
                S->elements[ri].length = i - S->elements[ri].start;
                S->length++;
            }
            return true;
        }
        if ( i < S->elements[ri].start ) break;
        ri++;
    }
    return false;
}

//

bool
int_set_remove_range(
    int_set_ref S,
    int_range_t r
)
{
    bool        rc = false;
    
    while ( r.length-- ) if ( int_set_remove_int(S, r.start++) ) rc = true;
    return rc;
}

//

bool
int_set_peek_next_int(
    int_set_ref S,
    int         *i
)
{
    if ( S->length ) {
        *i = S->elements[0].start;
        return true;
    }
    return false;
}

//

bool
int_set_pop_next_int(
    int_set_ref S,
    int         *i
)
{
    if ( S->length ) {
        *i = S->elements[0].start++, S->elements[0].length--;
        if ( S->elements[0].length == 0 ) {
            if ( S->length > 1 )
                memmove(&S->elements[0], &S->elements[1], sizeof(int_range_t) * (S->length - 1));
            S->length--;
        }
        return true;
    }
    return false;
}

//

void
int_set_summary(
    int_set_ref S,
    FILE        *stream
)
{
    int         ri = 0, counter = 0;
    
    fprintf(stream, "int_set@%p (l=%d, c=%d) {", S, S->length, S->capacity);
    while ( ri < S->length ) {
        fprintf(stream, (counter == 0) ? "\n    [%d, %d]" : ", [%d, %d]",
                    S->elements[ri].start, S->elements[ri].start + S->elements[ri].length - 1);
        counter = (counter + 1) % 16;
        ri++;
    }
    fprintf(stream, "\n}\n");
}

//
////
//

#ifdef ENABLE_INT_SET_TEST

int
main()
{
    int_set_ref     S = int_set_create();
    int             i;
    
    int_set_push_int(S, 10);
    int_set_summary(S, stdout);
    int_set_push_int(S, 11);
    int_set_summary(S, stdout);
    int_set_push_int(S, 12);
    int_set_summary(S, stdout);
    int_set_push_int(S, 15);
    int_set_summary(S, stdout);
    int_set_push_range(S, int_range_make_with_low_and_high(13, 14));
    int_set_summary(S, stdout);
    int_set_remove_range(S, int_range_make_with_low_and_high(14, 18));
    int_set_summary(S, stdout);
    int_set_remove_int(S, 11);
    int_set_summary(S, stdout);
    
    while ( int_set_pop_next_int(S, &i) ) printf("...%d...\n", i);

    int_set_destroy(S);

    return 0;
}

#endif
