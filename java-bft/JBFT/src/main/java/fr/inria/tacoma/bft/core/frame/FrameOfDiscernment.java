package fr.inria.tacoma.bft.core.frame;

import fr.inria.tacoma.bft.core.frame.impl.BigFrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.impl.SmallFrameOfDiscernment;

import java.util.*;
import java.util.stream.Collectors;

public abstract class FrameOfDiscernment {

    public static final int MAX_SMALL_FRAME_SIZE = 64;
    private final List<String> states;
    private final String name;
    private final List<StateSet> stateSets;

    public static FrameOfDiscernment newFrame(String name, String... elements) {
        if(elements.length > MAX_SMALL_FRAME_SIZE) {
            return new BigFrameOfDiscernment(name, elements);
        }
        else {
            return new SmallFrameOfDiscernment(name, elements);
        }
    }

    public static FrameOfDiscernment newFrame(String name, List<String> elements) {
        if(elements.size() > MAX_SMALL_FRAME_SIZE) {
            return new BigFrameOfDiscernment(name, elements);
        }
        else {
            return new SmallFrameOfDiscernment(name, elements);
        }
    }

    /**
     * Creates a new frame of discernment containing the elements given as
     * arguments. This implementation can only handle up to 64 elements.
     *
     * @param name the name of this frame of discernment.
     * @param elements strings describing the frame of discernment.
     * @throws IllegalArgumentException if there are more than 64 elements.
     */
    protected FrameOfDiscernment(String name, String... elements) {
        List<String> states = Arrays.asList(elements);
        Collections.sort(states);
        this.states = Collections.unmodifiableList(states);
        this.stateSets = this.states.stream()
                .map(this::toStateSet).collect(Collectors.toList());
        this.name = name;
    }

    protected FrameOfDiscernment(String name, List<String> elements) {
        List<String> elemList = new ArrayList<>(elements);
        Collections.sort(elemList);
        this.states = Collections.unmodifiableList(elemList);
        this.stateSets = this.states.stream()
                .map(this::toStateSet).collect(Collectors.toList());
        this.name = name;
    }

    /**
     * @return the list of possible states in this frame of discernment.
     */
    public List<String> getStates() {
        return this.states;
    }

    /**
     * @return the name of the frame of discernment
     */
    public String getName() {
        return this.name;
    }

    /**
     * Checks that the list of possible states in the frame of discernment
     * contains every parameter
     * @param elements array of state
     * @return true if all elements are in the frame, false otherwise
     */
    public boolean containsAll(String... elements) {
        return this.containsAll(Arrays.asList(elements));
    }


    /**
     * Checks that the list of possible states in the frame of discernment
     * contains every parameter
     * @param elements collection of state
     * @return true if all elements are in the frame, false otherwise
     */
    public boolean containsAll(Collection<String> elements) {
        return this.states.containsAll(elements);
    }

    public int card() {
        return this.states.size();
    }

    /**
     * Transforms a collection of string representing our states to a StateSet
     * object. Note that a StateSet is an immutable object so if you have to
     * reuse the same StateSet several time, avoid using toStateSet each time
     * and use the value return by previous computation of toStateSet.
     * @param elements collection containing every state in our state set as a string.
     * @return a new StateSet
     */
    public abstract StateSet toStateSet(Collection<String> elements);

    /**
     * Transforms an array of string representing our states to a StateSet
     * object.
     * @param elements array containing every state in our state set as a string.
     * @return a new StateSet
     */
    public StateSet toStateSet(String... elements) {
        return this.toStateSet(Arrays.asList(elements));
    }

    /**
     * Ths method returns the state set which contains every possible state.
     * This state set means that our world might be in every possible state
     * which in other word means that we are completely ignorant of what is
     * happening.
     * @return the set for full ignorance.
     */
    public abstract StateSet fullIgnoranceSet();

    /**
     * @return The empty state set for this frame of discernment
     */
    public abstract StateSet emptyStateSet();


    /**
     * Computes the set of state combinations with the given card
     * @param card number of states in the states combinations
     * @return the set f every combinations containing card states.
     */
    public Set<StateSet> getStateSetsWithCard(int card) {
        if(card < 0 || card > this.card()) {
            throw new IllegalArgumentException("Card" + card + " is not within bounds.");
        }
        if(card == this.card()) {
            Set<StateSet> set = new HashSet<>();
            set.add(this.fullIgnoranceSet());
            return set;
        }

        return recursiveCombinations(this.stateSets, this.emptyStateSet(), card, new HashSet<>());
    }

    private Set<StateSet> recursiveCombinations(List<StateSet> stateSets, StateSet currentSet,
                                                 int targetCard, Set<StateSet> accumulator) {

        if(currentSet.card() == targetCard) {
            //we created a set with the card we wanted, add it to the resulting list
            accumulator.add(currentSet);
        }

        for (int i = 0; i < stateSets.size(); i++) {

            StateSet stateSet = stateSets.get(i);
            List<StateSet> remaining = stateSets.subList(i +1, stateSets.size());
            StateSet partial_rec = currentSet.disjunction(stateSet);

            recursiveCombinations(remaining, partial_rec, targetCard, accumulator);
        }

        return accumulator;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        FrameOfDiscernment that = (FrameOfDiscernment) o;

        return this.states.equals(that.states) && this.name.equals(that.name);

    }

    @Override
    public int hashCode() {
        return this.states.hashCode();
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("{\"name\": \"")
                .append(this.name)
                .append("\", \"states\": ");
        String jsonList = this.states.stream()
                .map(str -> '"' + str + '"')
                .collect(Collectors.toList()).toString();
        builder.append(jsonList)
                .append('}');

        return   builder.toString();
    }
}
