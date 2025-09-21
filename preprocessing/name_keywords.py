drug_name_keywords = {
	# antibiotics – Treat suspected infection. Early use signals sepsis risk → mortality/LOS.
	"antibiotics": [ 
		# beta-lactams / combos
		"piperacillin", "tazobactam", "zosyn", "piperacillin tazobactam", "amoxicillin", "clavulanate", "amoxicillin clavulanate", "augmentin", "amox clav", "amox-clav", "ampicillin", "unasyn", "cefazolin", "ceftriaxone", "cefepime", "cephalexin", "keflex",
		# carbapenems / monobactams
		"meropenem", "imipenem", "ertapenem", "aztreonam",
		# aminoglycosides
		"gentamicin", "tobramycin", "amikacin",
		# glycopeptides / oxazolidinones / lipopeptides
		"vancomycin", "linezolid", "daptomycin",
		# quinolones
		"ciprofloxacin", "levofloxacin", "moxifloxacin",
		# sulfonamides + others
		"bactrim", "trimethoprim", "sulfamethoxazole", "tmp smx", "smx tmp", "cotrimoxazole", "co trimoxazole", "metronidazole", "clindamycin", "azithromycin"],

	# Airway/BP/ICU intensity
	# vasopressors – Support blood pressure in shock (norepi, etc.). Strong severity marker → mortality/LOS.
	"vasopressors": ["norepinephrine", "levophed", "epinephrine", "adrenaline", "phenylephrine", "neosynephrine", "dopamine", "dobutamine", "vasopressin", "milrinone"],
	
	# sedatives – Procedural/ICU sedation (propofol, dexmedetomidine). Indicates ventilation/procedures → severity.
	"sedatives": ["propofol", "midazolam", "lorazepam", "diazepam", "dexmedetomidine", "ketamine", "etomidate"],
	
	# opioids – Analgesia (fentanyl, morphine). Pain control; can correlate with ventilation or post-op status.
	"opioids": ["fentanyl", "morphine", "hydromorphone", "dilaudid", "oxycodone", "oxycontin", "percocet"],
	
	# bronchodilators – Albuterol/ipratropium. Respiratory distress/COPD exacerbation.
	"bronchodilators": ["albuterol", "salbutamol", "ipratropium", "duoneb", "combivent"],

	# Metabolic / endocrine
	# insulin – Glycemic control (IV/SC). Metabolic instability; ICU intensity.
	"insulin": ["insulin", "regular insulin", "aspart", "lispro", "glargine", "detemir", "novolog", "humalog", "lantus", "levemir", "humulin", "novolin", "nph"],
	
	# glucose_correction – D50/glucagon for hypoglycemia. Acute metabolic events.
	"glucose_correction": ["dextrose 50", "d50", "d25", "d10", "glucagon"],
	
	# electrolyte_repletion – K/Mg/Ca/PO₄ replacement. Metabolic instability → acuity/LOS.
	"electrolyte_repletion": ["potassium chloride", "kcl", "magnesium sulfate", "mgso4", "calcium gluconate", "sodium phosphate", "potassium phosphate", "neutra-phos", "neutra phos"],

	# Coagulation / cardio
	# anticoagulants – VTE/ACS treatment/prophylaxis. Thrombotic risk; bleeding trade-offs.
	"anticoagulants": ["heparin", "enoxaparin", "lovenox", "dalteparin", "tinzaparin", "warfarin", "coumadin", "dabigatran", "apixaban", "rivaroxaban"],
	
	# antiplatelets – ACS/stents; cardiology context.
	"antiplatelets": ["clopidogrel", "plavix", "ticagrelor", "prasugrel"],
	
	# beta_blockers – Rate/BP control. Cardiovascular comorbidity signal.
	"beta_blockers": ["metoprolol", "propranolol", "atenolol", "carvedilol", "bisoprolol", "esmolol", "labetalol", "nadolol", "sotalol"],
	
	# ace_inhibitors / arbs – Chronic BP/renal/cardiac care; less acute but useful comorbidity context.
	"ace_inhibitors": ["lisinopril", "enalapril", "captopril", "ramipril","perindopril", "quinapril", "benazepril"],
	
	"arbs": ["losartan", "valsartan", "irbesartan", "candesartan", "olmesartan"],
	
	# antihypertensives – Hydralazine. Acute BP management.
	"antihypertensives": ["hydralazine", "clonidine"],
	
	# rate_control_ccb – Diltiazem. AFib/tachyarrhythmia control.
	"rate_control_ccb": ["diltiazem", "cardizem", "verapamil"],
	
	# antiarrhythmics – Rhythm control in unstable arrhythmias; acute cardiac events.
	"antiarrhythmics": ["amiodarone", "lidocaine"],
	
	# vasodilators_nitrates – Afterload reduction/ischemia; acute cardiac management.
	"vasodilators_nitrates": ["nitroglycerin", "nitroprusside", "isosorbide dinitrate", "isosorbide mononitrate"],

	# Inflammation / immune / antifungals
	# steroids – Shock, COPD/asthma, cerebral edema. Severity/inflammation marker.
	"steroids": ["hydrocortisone", "methylprednisolone", "prednisone", "prednisolone", "dexamethasone"],
	
	# antifungals – Severe/opp. infections; strong severity signal.
	"antifungals": ["fluconazole", "voriconazole", "micafungin", "caspofungin", "amphotericin"],

	# Neuro
	# antipsychotics – Haloperidol. Delirium/ICU agitation management.
	"antipsychotics": ["haloperidol", "haldol", "quetiapine", "seroquel", "olanzapine", "zyprexa"],
	
	# anticonvulsants – Seizure control; neuro ICU acuity.
	"anticonvulsants": ["levetiracetam", "keppra", "valproate", "valproic acid", "depakote", "phenytoin", "dilantin"],
	
	# hypnotics – Zolpidem. Sleep aid; less ICU-specific.
	"hypnotics": ["zolpidem", "ambien", "eszopiclone", "lunesta"],
	
	# analgesic_adjuvant – Gabapentin. Neuropathic pain; chronic comorbidity signal.
	"analgesic_adjuvant": ["gabapentin", "neurontin", "pregabalin", "lyrica"],

	# GI / comfort
	# antiemetics – Nausea control (ondansetron, metoclopramide). Less predictive, but reflects GI issues/chemo-like care.
	"antiemetics": ["ondansetron", "zofran", "metoclopramide", "reglan"],
	
	# gi_prophylaxis – PPI/H2/antacids. Ventilator/ICU stress-ulcer prophylaxis; ICU care intensity.
	"gi_prophylaxis": ["pantoprazole", "omeprazole", "prilosec", "lansoprazole","ranitidine", "zantac", "famotidine", "pepcid", "calcium carbonate", "tums"],
	
	# laxatives – Bowel regimen. Low signal alone; part of ICU routine.
	"laxatives": ["docusate", "bisacodyl", "senna", "sennosides", "polyethylene glycol", "miralax", "lactulose", "milk of magnesia", "magnesium hydroxide"],
	
	# analgesic_antipyretic – Acetaminophen. Fever/pain control; nonspecific but common.
	"analgesic_antipyretic": ["acetaminophen", "paracetamol", "tylenol"],
	
	# analgesic_nsaid – Ketorolac. Non-opioid pain; renal/bleeding caution context.
	"analgesic_nsaid": ["ketorolac", "toradol"],

	# Chronic prevention context
	# cardio_preventive – Aspirin/statins. Chronic CAD prevention; comorbidity context.
	"cardio_preventive": ["aspirin", "aspirin ec", "atorvastatin", "simvastatin", "rosuvastatin"]
}


bio_specimen_keywords = [
	("blood",         r"\bblood|bld|line|catheter|cvc|peripheral"),
	("urine",         r"\burine|urinal|catheter urine|\bu[a]?\b"),
	("respiratory",   r"\bresp|sputum|trach|bronch|endotr|mini-?bal|bal|resp|endotr"),
	("wound",         r"\bwound|pus|abscess|swab|tissue(?!.*bone)|drain"),  # soft tissue/wound swabs
	("gi",            r"\bstool|gastric\s+aspirate|feces|rectal"),
	("hepatobiliary", r"\bbile|biliary"),
	("ent",           r"\bthroat|ear|nasal|np\s?swab|orophar|np/oph"),
	("eye",           r"\beye|corneal|conjunctival|ocular"),
	("skin",          r"\bskin\s*scraping|skin\s*scrapings|dermat"),
	("tissue_bone",   r"\bbiopsy|bone\s*marrow(?!.*cytogenetics)"),
	("marrow_cyto",   r"\bbone\s*marrow\s*-\s*cytogenetics"),
	("device",        r"\bforeign\s*body|hardware|implant|prosthesis"),
	("screening",     r"\bmrsa\s*screen|cre\s*screen|screen(ing)?"),
	("parasite",      r"\bworm|scotch\s*tape\s*prep|paddle|ova|parasite"),
	("internal fluids", r"\bfluid"),
	("herpes", r"\bherpes"), 
	("immunology", r"\bimmunology"), 
	("other",         r"\bxxx\b"),
]

common_organisms_columns = ['is_escherichia coli','is_staph aureus coag +','is_klebsiella pneumoniae','is_staphylococcus, coagulase negative','is_pseudomonas aeruginosa','is_proteus mirabilis','is_enterococcus sp.','is_yeast','is_enterobacter cloacae','is_klebsiella oxytoca','is_serratia marcescens','is_positive for methicillin resistant staph aureus','is_gram negative rod(s)','is_streptococcus pneumoniae','is_non-fermenter, not pseudomonas aeruginosa','is_citrobacter freundii complex','is_enterococcus faecalis','is_corynebacterium species (diphtheroids)','is_beta streptococcus group b','is_enterobacter aerogenes','is_enterococcus faecium','is_viridans streptococci','is_gram positive bacteria','is_acinetobacter baumannii','is_morganella morganii','is_clostridium difficile','is_staphylococcus epidermidis','is_acinetobacter baumannii complex','is_citrobacter koseri','is_2nd isolate','is_gram negative rod #2','is_bacteroides fragilis group','is_probable enterococcus','is_enterobacter cloacae complex','is_providencia stuartii','is_hafnia alvei','is_haemophilus influenzae, beta-lactamase negative','is_stenotrophomonas (xanthomonas) maltophilia','is_streptococcus anginosus (milleri) group','is_beta streptococcus group a','is_lactobacillus species','is_proteus vulgaris','is_candida albicans, presumptive identification','is_beta streptococci, not group a','is_gram negative rod #1','is_alpha streptococci','is_gram positive coccus(cocci)']

def get_keywords_cat():
	return drug_name_keywords, bio_specimen_keywords, common_organisms_columns