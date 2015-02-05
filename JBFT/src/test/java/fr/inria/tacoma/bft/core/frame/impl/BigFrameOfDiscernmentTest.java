package fr.inria.tacoma.bft.core.frame.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

public class BigFrameOfDiscernmentTest {

    private FrameOfDiscernment frame;

    @Before
    public void setUp() {
        this.frame = FrameOfDiscernment.newFrame("unittest", generateElements());
    }

    private List<String> generateElements() {
        List<String> elements = new ArrayList<>(100);
        for (int i = 0; i < 100; i++) {
            elements.add("" + i);
        }
        return elements;
    }

    @Test
    public void frame_emptyStateSet_ContainsNoSet() {
        StateSet fullIgnorance = frame.emptyStateSet();
        assertTrue(frame.getStates().stream().allMatch(set ->
                !fullIgnorance.includesOrEquals(frame.toStateSet(set))));
    }

    @Test
    public void frame_fullIgnoranceSet_ContainsEveryState() {
        StateSet fullIgnorance = frame.fullIgnoranceSet();
        assertTrue(frame.getStates().stream().allMatch(set ->
                fullIgnorance.includesOrEquals(frame.toStateSet(set))));
    }

    @Test
    public void StateSet_inclureOrEqualsWithIncludedSet_ReturnsTrue() {
        assertTrue(frame.toStateSet("1","2","49").includesOrEquals(frame.toStateSet("1","49")));
    }

    @Test
    public void StateSet_includeOrEqualsWithNonIncludedSet_ReturnsFalse() {
        assertFalse(frame.toStateSet("1", "2", "49").includesOrEquals(frame.toStateSet("5", "50")));
    }

    @Test
    public void StateSet_Conjunction_ReturnsRightResult() {
        StateSet conjunction = frame.toStateSet("1","2","49")
                .conjunction(frame.toStateSet("2", "49", "50"));
        assertEquals(frame.toStateSet("2", "49"), conjunction);
    }

    @Test
    public void StateSet_Disjunction_ReturnsRightResult() {
        StateSet disjunction = frame.toStateSet("1", "2", "49")
                .disjunction(frame.toStateSet("2", "49", "50"));
        assertEquals(frame.toStateSet("1", "2", "49", "50"), disjunction);
    }

    @Test
    public void toStringSet_OnStateSet_ReturnsTheRightSet() {
        StateSet set = frame.toStateSet("1", "2", "49");
        Set<String> expected = new HashSet<>(Arrays.asList("1", "2", "49"));
        assertEquals(expected, set.toStringSet());
    }

    @Test
    public void toStringSet_OnStateSet_ReturnsTheRightSet2() {
        StateSet set = frame.toStateSet("1", "2", "49", "25", "72");
        Set<String> expected = new HashSet<>(Arrays.asList("1", "2", "49", "25", "72"));
        assertEquals(expected, set.toStringSet());
    }

    @Test
    public void Frame_toString_returnsValidJson() {
        String content = frame.toString();
        try {
            new ObjectMapper().readTree(content);
        } catch (IOException e) {
            fail("Not a valid json string: " + content);
        }
    }

    @Test
    public void stateSet_CompareTo_IsGreaterThan0WithLowerCards() {
        int compareResult = frame.toStateSet("1", "2", "3").compareTo(frame.toStateSet("1", "2"));
        assertTrue(compareResult > 0);
        compareResult = frame.toStateSet("1", "2").compareTo(frame.toStateSet("1"));
        assertTrue(compareResult > 0);
        compareResult = frame.toStateSet("1", "4", "5").compareTo(frame.toStateSet("2"));
        assertTrue(compareResult > 0);
    }

    @Test
    public void stateSet_CompareTo_IsLessThan0WithLowerCards() {
        int compareResult = frame.toStateSet("1", "2").compareTo(frame.toStateSet("1", "2", "3"));
        assertTrue(compareResult < 0);
        compareResult = frame.toStateSet("2").compareTo(frame.toStateSet("1", "4", "5"));
        assertTrue(compareResult < 0);
    }

    @Test
    public void stateSet_CompareTo_IsConsistentWithEquals() {
        StateSet stateSet = frame.toStateSet("1", "2", "3");
        assertEquals(0, stateSet.compareTo(stateSet));
        assertNotEquals(0, stateSet.compareTo(frame.toStateSet("2", "4", "5")));
    }

    @Test
    public void stateSet_CompareTo_IsTransitive() {
        StateSet stateA = frame.toStateSet("1");
        StateSet stateB = frame.toStateSet("2");
        StateSet stateC = frame.toStateSet("3");
        assertTrue("state 1 should be before 2", stateA.compareTo(stateB) < 0);
        assertTrue("state 2 should be before 3", stateB.compareTo(stateC) < 0);
        assertTrue("state 1 should be before 3", stateA.compareTo(stateC) < 0);
    }


    @Test
    public void stateSet_difference_ReturnsRightResult() {
        assertEquals(frame.toStateSet("1"),
                     frame.toStateSet("1", "2").difference(frame.toStateSet("2")));
    }

    @Test
    public void stateSet_difference_ReturnsRightResult2() {
        assertEquals(frame.toStateSet("1"),
                     frame.toStateSet("1", "2").difference(frame.toStateSet("2", "3")));
    }

}
