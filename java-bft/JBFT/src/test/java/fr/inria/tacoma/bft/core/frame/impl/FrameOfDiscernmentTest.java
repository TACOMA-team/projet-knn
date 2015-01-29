package fr.inria.tacoma.bft.core.frame.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.Assert.*;


public class FrameOfDiscernmentTest {

    private FrameOfDiscernment frame;

    @Before
    public void setUp() throws Exception {
        frame = FrameOfDiscernment.newFrame("unittest", "a", "b", "c", "d", "e");
    }

    @Test
    public void card_AfterCallingConstructorWith5Args_return5() {
        assertEquals(5, frame.card());
    }

    @Test
    public void getStateSets_WithCard1_ReturnsEveryState() {
        Set<StateSet> expected = frame.getStates().stream()
                .map(frame::toStateSet).collect(Collectors.toSet());
        assertEquals(expected, frame.getStateSetsWithCard(1));
    }

    @Test
    public void getStateSets_WithCardEqualFrameSize_ReturnsFullIgnoranceSet() {
        Set<StateSet> expected =
                new HashSet<>(Arrays.asList(new StateSet[] {frame.fullIgnoranceSet()}));
        assertEquals(expected, frame.getStateSetsWithCard(frame.card()));
    }

    @Test
    public void getStateSets_WithCardEqual2_ReturnEveryCombinationWith2Elements() {
        Set<StateSet> expected = new HashSet<>();
        expected.add(frame.toStateSet("a", "b"));
        expected.add(frame.toStateSet("a", "c"));
        expected.add(frame.toStateSet("a", "d"));
        expected.add(frame.toStateSet("a", "e"));
        expected.add(frame.toStateSet("b", "c"));
        expected.add(frame.toStateSet("b", "d"));
        expected.add(frame.toStateSet("b", "e"));
        expected.add(frame.toStateSet("c", "d"));
        expected.add(frame.toStateSet("c", "e"));
        expected.add(frame.toStateSet("d", "e"));
        assertEquals(expected, frame.getStateSetsWithCard(2));
    }

    @Test
    public void getStateSets_WithCardEqual3_ReturnEveryCombinationWith3Elements() {
        Set<StateSet> expected = new HashSet<>();
        expected.add(frame.toStateSet("a", "b", "c"));
        expected.add(frame.toStateSet("a", "b", "d"));
        expected.add(frame.toStateSet("a", "b", "e"));
        expected.add(frame.toStateSet("a", "c", "d"));
        expected.add(frame.toStateSet("a", "c", "e"));
        expected.add(frame.toStateSet("a", "d", "e"));
        expected.add(frame.toStateSet("b", "c", "d"));
        expected.add(frame.toStateSet("b", "c", "e"));
        expected.add(frame.toStateSet("b", "d", "e"));
        expected.add(frame.toStateSet("c", "d", "e"));
        assertEquals(expected, frame.getStateSetsWithCard(3));
    }

    @Test
    public void toStringSet_OnStateSet_ReturnsTheRightSet() {
        StateSet set = frame.toStateSet("d", "e", "b");
        Set<String> expected = new HashSet<>(Arrays.asList("d", "e", "b"));
        assertEquals(expected, set.toStringSet());
    }

    @Test
    public void toStringSet_OnStateSet_ReturnsTheRightSet2() {
        StateSet set = frame.toStateSet("d", "e", "b","a");
        Set<String> expected = new HashSet<>(Arrays.asList("d", "e", "b","a"));
        assertEquals(expected, set.toStringSet());
    }

    @Test
    public void toString_OnElementSet_ShouldPrintTheRightSet() {
        StateSet set = frame.toStateSet("d", "e", "b");
        assertEquals("[\"b\", \"d\", \"e\"]",set.toString());
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
    public void stateSet_difference_ReturnsRightResult() {
        assertEquals(frame.toStateSet("a"),
                     frame.toStateSet("a", "b").difference(frame.toStateSet("b")));
    }

    @Test
      public void stateSet_difference_ReturnsRightResult2() {
        assertEquals(frame.toStateSet("a"),
                     frame.toStateSet("a", "b").difference(frame.toStateSet("b", "c")));
    }


    @Test
    public void stateSet_complement_ReturnsRightResult() {
        assertEquals(frame.toStateSet("c","d","e"), frame.toStateSet("a", "b").complement());
    }

    @Test
    public void stateSet_CompareTo_IsGreaterThan0WithLowerCards() {
        int compareResult = frame.toStateSet("a", "b", "c").compareTo(frame.toStateSet("a", "b"));
        assertTrue(compareResult > 0);
        compareResult = frame.toStateSet("a", "b").compareTo(frame.toStateSet("a"));
        assertTrue(compareResult > 0);
        compareResult = frame.toStateSet("a", "d", "e").compareTo(frame.toStateSet("b"));
        assertTrue(compareResult > 0);
    }

    @Test
    public void stateSet_CompareTo_IsLessThan0WithLowerCards() {
        int compareResult = frame.toStateSet("a", "b").compareTo(frame.toStateSet("a", "b", "c"));
        assertTrue(compareResult < 0);
        compareResult = frame.toStateSet("b").compareTo(frame.toStateSet("a", "d", "e"));
        assertTrue(compareResult < 0);
    }

    @Test
    public void stateSet_CompareTo_IsConsistentWithEquals() {
        StateSet stateSet = frame.toStateSet("a", "b", "c");
        assertEquals(0, stateSet.compareTo(stateSet));
        assertNotEquals(0, stateSet.compareTo(frame.toStateSet("c", "d", "e")));
    }

    @Test
     public void stateSet_CompareTo_IsTransitive() {
        StateSet stateA = frame.toStateSet("a");
        StateSet stateB = frame.toStateSet("b");
        StateSet stateC = frame.toStateSet("c");
        assertTrue("state a should be before b", stateA.compareTo(stateB) < 0);
        assertTrue("state b should be before c", stateB.compareTo(stateC) < 0);
        assertTrue("state a should be before c", stateA.compareTo(stateC) < 0);
    }
}
