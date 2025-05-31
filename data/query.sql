-- sqllite query to be used by online csv to sql tool 

 SELECT 
    S1.rush_att AS team1_rush_att,
    S1.rush_yds AS team1_rush_yds,
    S1.rush_tds AS team1_rush_tds,
    S1.pass_cmp AS team1_pass_cmp,
    S1.pass_att AS team1_pass_att,
    S1.pass_cmp_pct AS team1_pass_cmp_pct,
    S1.pass_yds AS team1_pass_yds,
    S1.pass_tds AS team1_pass_tds,
    S1.pass_int AS team1_pass_int,
    S1.passer_rating AS team1_passer_rating,
    S1.net_pass_yds AS team1_net_pass_yds,
    S1.total_yds AS team1_total_yds,
    S1.times_sacked AS team1_times_sacked,
    S1.yds_sacked_for AS team1_yds_sacked_for,
    S1.fumbles AS team1_fumbles,
    S1.fumbles_lost AS team1_fumbles_lost,
    S1.turnovers AS team1_turnovers,
    S1.penalties AS team1_penalties,
    S1.penalty_yds AS team1_penalty_yds,
    S1.first_downs AS team1_first_downs,
    S1.third_down_conv AS team1_third_down_conv,
    S1.third_down_att AS team1_third_down_att,
    S1.third_down_conv_pct AS team1_third_down_conv_pct,
    S1.fourth_down_conv AS team1_fourth_down_conv,
    S1.fourth_down_att AS team1_fourth_down_att,
    S1.fourth_down_conv_pct AS team1_fourth_down_conv_pct,
    S1.time_of_possession AS team1_time_of_possession,

    -- Team 2 Statistics 
    S2.rush_att AS team2_rush_att,
    S2.rush_yds AS team2_rush_yds,
    S2.rush_tds AS team2_rush_tds,
    S2.pass_cmp AS team2_pass_cmp,
    S2.pass_att AS team2_pass_att,
    S2.pass_cmp_pct AS team2_pass_cmp_pct,
    S2.pass_yds AS team2_pass_yds,
    S2.pass_tds AS team2_pass_tds,
    S2.pass_int AS team2_pass_int,
    S2.passer_rating AS team2_passer_rating,
    S2.net_pass_yds AS team2_net_pass_yds,
    S2.total_yds AS team2_total_yds,
    S2.times_sacked AS team2_times_sacked,
    S2.yds_sacked_for AS team2_yds_sacked_for,
    S2.fumbles AS team2_fumbles,
    S2.fumbles_lost AS team2_fumbles_lost,
    S2.turnovers AS team2_turnovers,
    S2.penalties AS team2_penalties,
    S2.penalty_yds AS team2_penalty_yds,
    S2.first_downs AS team2_first_downs,
    S2.third_down_conv AS team2_third_down_conv,
    S2.third_down_att AS team2_third_down_att,
    S2.third_down_conv_pct AS team2_third_down_conv_pct,
    S2.fourth_down_conv AS team2_fourth_down_conv,
    S2.fourth_down_att AS team2_fourth_down_att,
    S2.fourth_down_conv_pct AS team2_fourth_down_conv_pct,
    S2.time_of_possession AS team2_time_of_possession,

    -- Dependent variable: team1_win (1 if team1_nano won, 0 otherwise)
    -- As per the paper, 'tm_nano' in Season-2021 is always the winning team.
    -- Ties are counted as 'non-wins' (0). [cite: 727]
    CASE
        WHEN SS.tm_nano = S1.nano THEN 1
        ELSE 0
    END AS team1_win

FROM
    "Stats" AS S1
JOIN
    "Stats" AS S2
    ON S1.boxscore_stats_link = S2.boxscore_stats_link
    AND S1.nano < S2.nano -- Ensures unique pairs and consistent team1/team2 assignment based on nano alphabetical order
JOIN
    "Season" AS SS
    ON S1.boxscore_stats_link = SS.boxscore_stats_link;