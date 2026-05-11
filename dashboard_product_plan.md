# Predictive Modeling Dashboard Plan: "IntentSight"

## 1. Product Vision & Aesthetics
The goal is to deliver a **high-fidelity, minimalist, visual-first UI** that transforms dense machine learning inferences into immediate, actionable insights. The design language utilizes a sleek dark-mode aesthetic with "glassmorphism" components to maintain data density without feeling cluttered. 

![Dashboard High-Fidelity Mockup](C:\Users\tzx20\.gemini\antigravity\brain\e596166e-b516-40ee-b4d2-baa25fe0ad36\artifacts\dashboard_planning_ui.png)
*Concept rendering of the visual-first dashboard interface.*

---

## 2. Core UI Layout & Components

### A. The Global View Toggle (Header)
At the very top of the interface, a prominent, pill-shaped toggle switch controls the granularity of the entire central visualization area. 
* **Aggregated:** Shows the macro-level predictive trend across all user demographics.
* **Category:** Splits the data into overarching groupings (e.g., By Gender, By Income Bracket).
* **Individual Type:** High-resolution view isolating specific intent predictions down to granular user cohorts.

### B. Central Line Charts (Absolute Monthly Trends)
To ensure the design is data-dense but highly legible, the central charts avoid complex standardizations and focus solely on **absolute monthly values and trends**.
* **Visuals:** Sleek, neon-accented bezier curves on a dark background grid.
* **Interactivity:** Hovering over any point triggers a custom tooltip identifier, immediately revealing the exact absolute volume and model confidence for that unique series in that month.

### C. Dynamic Multi-Select Filter Panel (Sidebar/Floating)
A non-intrusive filter menu allowing real-time slicing of the data. 
* **Models:** Toggle between `Tuned Extra Trees`, `Random Forest`, `LightGBM`, etc.
* **Categories:** Dynamically filter the predictions by Behavioral Metrics (App Usage Time, Match Rate, BMI).

### D. Real-Time Predictive Heat Map (Bottom/Split View)
A visually striking matrix visualization showing the correlation density between user behaviors and their predicted Intent Class over time.
* Colored using a continuous, minimalist gradient (e.g., deep indigo to vibrant cyan) indicating high vs. low absolute volume.
* Updates instantaneously as the multi-select filters are adjusted.

---

## 3. Technology Stack Recommendation
To achieve this specific high-fidelity look and real-time responsiveness:
* **Frontend Framework:** Next.js (React) or Vue 3
* **Styling:** Vanilla CSS with custom design tokens (or highly customized Tailwind configuration) specifically leveraging `backdrop-filter: blur()` for glassmorphism.
* **Data Visualization:** Recharts or Nivo (React) for highly customizable, SVG-based line charts and heatmaps with built-in hover tooltip support.
* **Backend Integration:** FastAPI serving the pickled `best_tuned_model.pkl` to provide real-time inference numbers.
