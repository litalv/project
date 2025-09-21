# icustays.icustay_id::INTEGER AS icustay_id, icustays.intime::DATE AS intime, icustays.outtime::DATE AS outtime
sqls = {
"ICUQ" : f"""--sql
			SELECT admissions.subject_id::INTEGER AS subject_id, 
				   admissions.hadm_id::INTEGER AS hadm_id,
				   admissions.admittime::DATE AS admittime, 
				   admissions.dischtime::DATE AS dischtime,
				   admissions.ethnicity, 
				   admissions.deathtime::DATE AS deathtime, 
				   patients.gender, 
				   patients.dob::DATE AS dob, 
				   patients.dod::DATE as dod
			FROM admissions
			INNER JOIN patients
				ON admissions.subject_id = patients.subject_id
			LEFT JOIN icustays
				ON admissions.hadm_id = icustays.hadm_id

			WHERE admissions.has_chartevents_data = 1
				AND admissions.subject_id::INTEGER IN ?
			ORDER BY admissions.subject_id, admissions.hadm_id, admissions.admittime;
			""",

"LABQUERY" : f"""--sql
				SELECT labevents.subject_id::INTEGER AS subject_id,
					   labevents.hadm_id::INTEGER AS hadm_id,
					   date_trunc('hour', CAST(labevents.charttime AS TIMESTAMP)) AS charttime,
					   labevents.itemid::INTEGER AS itemid,
					   labevents.valuenum::DOUBLE AS valuenum,
					   date_trunc('hour', CAST(admissions.admittime AS TIMESTAMP)) AS admittime
				FROM labevents
				  INNER JOIN admissions
							ON labevents.subject_id = admissions.subject_id
								AND labevents.hadm_id = admissions.hadm_id
								AND labevents.charttime::DATE between
									(admissions.admittime::DATE)
									AND (admissions.admittime::DATE + interval 48 hour)
								AND itemid::INTEGER IN ? \
								""",

# chartevents.charttime AS charttime,
# admissions.admittime AS admittime
"VITQUERY" : f"""--sql
				SELECT chartevents.subject_id::INTEGER AS subject_id,
					  chartevents.hadm_id::INTEGER AS hadm_id,
					  date_trunc('hour', CAST(chartevents.charttime AS TIMESTAMP)) AS charttime,
					  chartevents.itemid::INTEGER AS itemid,
					  chartevents.valuenum::DOUBLE AS valuenum,
					  date_trunc('hour', CAST(admissions.admittime AS TIMESTAMP)) AS admittime
				FROM chartevents
						 INNER JOIN admissions
									ON chartevents.subject_id = admissions.subject_id
										AND chartevents.hadm_id = admissions.hadm_id
										AND chartevents.charttime::DATE between
										   (admissions.admittime::DATE)
										   AND (admissions.admittime::DATE + interval 48 hour)
										AND itemid::INTEGER in ?
			  -- exclude rows marked as error
			  AND chartevents.error::INTEGER IS DISTINCT \
				FROM 1 \
				""",

"PRESCRIPTIONS" : f"""--sql
			SELECT a.subject_id::INTEGER AS subject_id, 
				a.hadm_id::INTEGER AS hadm_id,
				date_trunc('hour', CAST(p.startdate AS TIMESTAMP)) AS charttime,
				lower(coalesce(p.drug_name_generic, p.drug)) AS drug_name,
				p.route, p.dose_val_rx, p.dose_unit_rx
			FROM prescriptions p
			JOIN admissions a
				ON p.subject_id = a.subject_id
				AND p.hadm_id    = a.hadm_id
			WHERE coalesce(p.drug_name_generic, p.drug) IS NOT NULL
				AND p.startdate::DATE between
					(a.admittime::DATE) AND (a.admittime::DATE + interval 48 hour)
				AND a.subject_id::INTEGER IN ?
			ORDER BY a.subject_id, a.hadm_id, charttime;
			""",
			
"MICROBIOLOGY" : f"""--sql
			SELECT m.subject_id::INTEGER AS subject_id,
				m.hadm_id::INTEGER    AS hadm_id,
				date_trunc('hour', CAST(m.charttime AS TIMESTAMP)) AS charttime,
				lower(m.spec_itemid) AS specimen,  
				lower(m.SPEC_TYPE_DESC) AS specimen_type,
				lower(m.org_name) AS organism,
				lower(m.ab_name) AS antibiotic,
				m.interpretation,  -- 'P' (positive), 'S', 'R' etc. depending on field
			FROM microbiologyevents m
			JOIN admissions a
			ON m.subject_id = a.subject_id
				AND m.hadm_id = a.hadm_id
			WHERE m.charttime::DATE between (a.admittime::DATE) AND (a.admittime::DATE + interval 48 hour)
				AND a.subject_id::INTEGER IN ?
			ORDER BY a.subject_id, a.hadm_id, charttime;
			"""
}

def get_sql_queries():
	return sqls